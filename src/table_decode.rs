//! SLANet_plus structure-token decoding.
//!
//! [`argmax_tokens`] (greedy per-step argmax over the structure head, skipping
//! the start token and stopping at the end token) is translated from the
//! `recognize` loop in `src/rapid_table.rs` of `bobine` 0.7.3
//! (<https://github.com/opticsWolf/bobine>), used under its MIT option.
//! `bobine` only builds HTML from the tokens; [`decode_structure_tokens`], which
//! builds a span-aware logical grid, is new code.
//!
//! MIT License
//!
//! Copyright (c) 2026 opticsWolf
//!
//! Permission is hereby granted, free of charge, to any person obtaining a copy
//! of this software and associated documentation files (the "Software"), to deal
//! in the Software without restriction, including without limitation the rights
//! to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
//! copies of the Software, and to permit persons to whom the Software is
//! furnished to do so, subject to the following conditions:
//!
//! The above copyright notice and this permission notice shall be included in all
//! copies or substantial portions of the Software.
//!
//! THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
//! IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//! FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
//! AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//! LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
//! OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
//! SOFTWARE.

// Consumed by the table-recognition stage added in a later task.
#![allow(dead_code)]

use crate::orient::probabilities;
use ndarray::ArrayView2;

/// Largest `colspan`/`rowspan` accepted from the model. Larger values are
/// clamped and flagged as malformed, which bounds the occupancy bookkeeping.
const MAX_SPAN: usize = 64;

/// One decoded structure token: the step of the structure head it came from
/// (counting the skipped start token), the token text and its probability.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct StepToken {
    pub step: usize,
    pub token: String,
    pub prob: f32,
}

/// Greedy decoding of the structure head `[T, classes]`. `charset` has `sos`
/// first and `eos` last. Stops at `eos` after step 0, skips `sos`/`eos`, and
/// reads at most `max_tokens` steps.
pub(crate) fn argmax_tokens(
    structure: ArrayView2<f32>,
    charset: &[String],
    max_tokens: usize,
) -> Vec<StepToken> {
    let mut out = Vec::new();
    if charset.is_empty() {
        return out;
    }
    let eos = charset.len() - 1;
    let steps = structure.nrows().min(max_tokens);
    for t in 0..steps {
        let row: Vec<f32> = structure.row(t).to_vec();
        let mut idx = 0usize;
        let mut best = f32::NEG_INFINITY;
        for (c, &v) in row.iter().enumerate() {
            if v > best {
                best = v;
                idx = c;
            }
        }
        if t > 0 && idx == eos {
            break;
        }
        if idx == 0 || idx == eos {
            continue;
        }
        let Some(token) = charset.get(idx) else {
            continue;
        };
        let prob = probabilities(&row).get(idx).copied().unwrap_or(0.0);
        out.push(StepToken {
            step: t,
            token: token.clone(),
            prob,
        });
    }
    out
}

/// A logical table cell. `token_index` is the index in the token slice of the
/// token that opened the cell (`<td></td>` or `<td`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct GridCell {
    pub row: usize,
    pub col: usize,
    pub row_span: usize,
    pub col_span: usize,
    pub token_index: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub(crate) struct DecodedGrid {
    pub cells: Vec<GridCell>,
    pub num_rows: usize,
    pub num_cols: usize,
    pub malformed: bool,
}

#[derive(Default)]
struct Builder {
    row_open: bool,
    rows: usize,
    cursor: usize,
    occupied: Vec<Vec<bool>>,
    pending: Option<GridCell>,
    cells: Vec<GridCell>,
    malformed: bool,
}

impl Builder {
    fn ensure_row(&mut self) {
        if !self.row_open {
            self.malformed = true;
            self.rows += 1;
            self.row_open = true;
            self.cursor = 0;
        }
    }

    /// Place a cell in the current row at the first free column at or after
    /// the cursor, marking every grid slot it covers as occupied.
    fn place(&mut self, mut cell: GridCell) {
        let row = self.rows - 1;
        let (rs, cs) = (cell.row_span.max(1), cell.col_span.max(1));
        // First column at or after the cursor where the whole `cs` slots of the
        // current row are free (HTML table auto-layout). The scan may run past
        // the current row width, which extends the row.
        let slot_free = |occ: &Vec<Vec<bool>>, c: usize| {
            !occ.get(row)
                .and_then(|r| r.get(c))
                .copied()
                .unwrap_or(false)
        };
        let mut col = self.cursor;
        while !(col..col + cs).all(|c| slot_free(&self.occupied, c)) {
            col += 1;
        }
        if self.occupied.len() < row + rs {
            self.occupied.resize(row + rs, Vec::new());
        }
        for r in 0..rs {
            let line = &mut self.occupied[row + r];
            if line.len() < col + cs {
                line.resize(col + cs, false);
            }
            for slot in &mut line[col..col + cs] {
                *slot = true;
            }
        }
        cell.row = row;
        cell.col = col;
        self.cursor = col + cs;
        self.cells.push(cell);
    }

    fn finish_pending(&mut self) {
        if let Some(p) = self.pending.take() {
            self.malformed = true;
            self.place(p);
        }
    }

    fn set_span(&mut self, value: &str, is_col: bool) {
        let parsed = value.parse::<usize>().ok().filter(|&n| n >= 1);
        let Some(p) = self.pending.as_mut() else {
            self.malformed = true;
            return;
        };
        let Some(n) = parsed else {
            self.malformed = true;
            return;
        };
        let n = if n > MAX_SPAN {
            self.malformed = true;
            MAX_SPAN
        } else {
            n
        };
        if is_col {
            p.col_span = n;
        } else {
            p.row_span = n;
        }
    }
}

fn new_cell(i: usize) -> GridCell {
    GridCell {
        row: 0,
        col: 0,
        row_span: 1,
        col_span: 1,
        token_index: i,
    }
}

/// Decode SLANet_plus structure tokens into a logical grid with spans.
/// Never panics; malformed input sets `malformed` and returns the best-effort grid.
pub(crate) fn decode_structure_tokens(tokens: &[&str]) -> DecodedGrid {
    let mut b = Builder::default();
    for (i, &tok) in tokens.iter().enumerate() {
        match tok {
            "<thead>" | "</thead>" | "<tbody>" | "</tbody>" => {}
            "<tr>" => {
                if b.row_open {
                    b.malformed = true;
                }
                b.finish_pending();
                b.rows += 1;
                b.row_open = true;
                b.cursor = 0;
            }
            "</tr>" => {
                b.finish_pending();
                if !b.row_open {
                    b.malformed = true;
                }
                b.row_open = false;
            }
            "<td></td>" => {
                b.finish_pending();
                b.ensure_row();
                b.place(new_cell(i));
            }
            "<td" => {
                b.finish_pending();
                b.ensure_row();
                b.pending = Some(new_cell(i));
            }
            ">" => {
                if b.pending.is_none() {
                    b.malformed = true;
                }
            }
            "</td>" => match b.pending.take() {
                Some(p) => b.place(p),
                None => b.malformed = true,
            },
            other => {
                let span =
                    |prefix: &str| other.strip_prefix(prefix).and_then(|s| s.strip_suffix('"'));
                if let Some(v) = span(" colspan=\"") {
                    b.set_span(v, true);
                } else if let Some(v) = span(" rowspan=\"") {
                    b.set_span(v, false);
                } else {
                    b.malformed = true;
                }
            }
        }
    }
    b.finish_pending();

    let num_rows = b.rows;
    let mut cells = b.cells;
    for c in &mut cells {
        c.row_span = c.row_span.min(num_rows - c.row).max(1);
    }
    let num_cols = cells.iter().map(|c| c.col + c.col_span).max().unwrap_or(0);
    DecodedGrid {
        cells,
        num_rows,
        num_cols,
        malformed: b.malformed,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cells(g: &DecodedGrid) -> Vec<(usize, usize, usize, usize)> {
        g.cells
            .iter()
            .map(|c| (c.row, c.col, c.row_span, c.col_span))
            .collect()
    }

    /// Rebuild an occupancy map from the cells and panic on a double claim.
    fn assert_no_overlap(g: &DecodedGrid) {
        let mut seen = std::collections::HashSet::new();
        for c in &g.cells {
            for r in c.row..c.row + c.row_span {
                for k in c.col..c.col + c.col_span {
                    assert!(seen.insert((r, k)), "slot ({r},{k}) claimed twice: {g:?}");
                }
            }
        }
    }
    const TD: &str = "<td></td>";

    #[test]
    fn decode_simple_two_by_two() {
        let g = decode_structure_tokens(&["<tr>", TD, TD, "</tr>", "<tr>", TD, TD, "</tr>"]);
        assert_eq!(
            cells(&g),
            vec![(0, 0, 1, 1), (0, 1, 1, 1), (1, 0, 1, 1), (1, 1, 1, 1)]
        );
        assert_eq!((g.num_rows, g.num_cols, g.malformed), (2, 2, false));
        assert_no_overlap(&g);
    }

    #[test]
    fn decode_fixture_3x4_with_merged_header() {
        let mut t = vec![
            "<thead>",
            "<tr>",
            "<td",
            " colspan=\"2\"",
            ">",
            "</td>",
            TD,
            TD,
            "</tr>",
            "</thead>",
            "<tbody>",
        ];
        for _ in 0..2 {
            t.extend(["<tr>", TD, TD, TD, TD, "</tr>"]);
        }
        t.push("</tbody>");
        let g = decode_structure_tokens(&t);
        assert_eq!(
            (g.num_rows, g.num_cols, g.cells.len(), g.malformed),
            (3, 4, 11, false)
        );
        assert_eq!(cells(&g)[0], (0, 0, 1, 2));
        assert_eq!(cells(&g)[1], (0, 2, 1, 1));
        assert_eq!(g.cells[0].token_index, 2);
        assert_no_overlap(&g);
    }

    #[test]
    fn decode_rowspan_pushes_next_row() {
        let g = decode_structure_tokens(&[
            "<tr>",
            "<td",
            " rowspan=\"2\"",
            ">",
            "</td>",
            TD,
            "</tr>",
            "<tr>",
            TD,
            "</tr>",
        ]);
        assert_eq!(cells(&g), vec![(0, 0, 2, 1), (0, 1, 1, 1), (1, 1, 1, 1)]);
        assert!(!g.malformed);
        assert_no_overlap(&g);
    }

    #[test]
    fn decode_rowspan_and_colspan_together() {
        let g = decode_structure_tokens(&[
            "<tr>",
            "<td",
            " colspan=\"2\"",
            " rowspan=\"2\"",
            ">",
            "</td>",
            TD,
            "</tr>",
            "<tr>",
            TD,
            "</tr>",
        ]);
        assert_eq!(cells(&g), vec![(0, 0, 2, 2), (0, 2, 1, 1), (1, 2, 1, 1)]);
        assert_eq!((g.num_rows, g.num_cols, g.malformed), (2, 3, false));
        assert_no_overlap(&g);
    }

    #[test]
    fn decode_colspan_wider_than_gap_skips_occupied_column() {
        // Row 1 col 1 is taken by the rowspan cell; a colspan=2 cell cannot
        // fit at col 0 and moves to cols 2-3.
        let g = decode_structure_tokens(&[
            "<tr>",
            TD,
            "<td",
            " rowspan=\"2\"",
            ">",
            "</td>",
            "</tr>",
            "<tr>",
            "<td",
            " colspan=\"2\"",
            ">",
            "</td>",
            "</tr>",
        ]);
        assert_eq!(cells(&g), vec![(0, 0, 1, 1), (0, 1, 2, 1), (1, 2, 1, 2)]);
        assert_eq!((g.num_rows, g.num_cols, g.malformed), (2, 4, false));
        assert_no_overlap(&g);
    }

    #[test]
    fn decode_single_branch_malformed_cases() {
        // `</td>` without pending `<td`
        let g = decode_structure_tokens(&["<tr>", "</td>", "</tr>"]);
        assert!(g.malformed && g.cells.is_empty());
        // unknown token
        let g = decode_structure_tokens(&["<tr>", "<b>", "</tr>"]);
        assert!(g.malformed && g.cells.is_empty());
        // `</tr>` with pending `<td`: cell kept
        let g = decode_structure_tokens(&["<tr>", "<td", "</tr>"]);
        assert!(g.malformed);
        assert_eq!(cells(&g), vec![(0, 0, 1, 1)]);
        // `<tr>` with pending `<td`: cell placed in the old row
        let g = decode_structure_tokens(&["<tr>", "<td", "<tr>", "</tr>"]);
        assert!(g.malformed);
        assert_eq!((g.num_rows, g.num_cols), (2, 1));
        assert_eq!(cells(&g), vec![(0, 0, 1, 1)]);
    }

    #[test]
    fn decode_rowspan_beyond_last_row_is_clamped() {
        let g = decode_structure_tokens(&["<tr>", "<td", " rowspan=\"5\"", ">", "</td>", "</tr>"]);
        assert_eq!(cells(&g), vec![(0, 0, 1, 1)]);
    }

    #[test]
    fn decode_malformed_sequence_does_not_panic() {
        let g = decode_structure_tokens(&[
            TD,
            "</td>",
            "<tr>",
            "<td",
            "<b>",
            TD,
            "<tr>",
            "<td",
            " colspan=\"x\"",
        ]);
        assert!(g.malformed);
        assert_eq!(
            cells(&g),
            vec![(0, 0, 1, 1), (1, 0, 1, 1), (1, 1, 1, 1), (2, 0, 1, 1)]
        );
        assert_eq!((g.num_rows, g.num_cols), (3, 2));
    }

    #[test]
    fn decode_individual_malformed_cases() {
        // `<tr>` inside `<tr>`
        assert!(decode_structure_tokens(&["<tr>", "<tr>", TD, "</tr>"]).malformed);
        // `</tr>` without `<tr>`
        assert!(decode_structure_tokens(&["</tr>"]).malformed);
        // span without a pending cell
        assert!(decode_structure_tokens(&["<tr>", " colspan=\"2\"", TD, "</tr>"]).malformed);
        // `>` without `<td`
        assert!(decode_structure_tokens(&["<tr>", ">", "</tr>"]).malformed);
        // oversized span is clamped and flagged, never allocates unboundedly
        let g = decode_structure_tokens(&[
            "<tr>",
            "<td",
            " colspan=\"999999999\"",
            " rowspan=\"999999999\"",
            ">",
            "</td>",
            "</tr>",
        ]);
        assert!(g.malformed);
        assert_eq!((g.num_rows, g.num_cols), (1, MAX_SPAN));
    }

    #[test]
    fn decode_unclosed_cell_at_end_is_malformed_but_kept() {
        let g = decode_structure_tokens(&["<tr>", "<td", " colspan=\"2\""]);
        assert!(g.malformed);
        assert_eq!(cells(&g), vec![(0, 0, 1, 2)]);
    }

    #[test]
    fn decode_empty_sequence() {
        assert_eq!(decode_structure_tokens(&[]), DecodedGrid::default());
    }

    #[test]
    fn decode_zero_cells_gives_empty_grid() {
        let g = decode_structure_tokens(&["<b>", "garbage", "</td>"]);
        assert!(g.malformed);
        assert_eq!((g.cells.len(), g.num_rows, g.num_cols), (0, 0, 0));
        let g = decode_structure_tokens(&["<tr>", "</tr>"]);
        assert_eq!(
            (g.cells.len(), g.num_rows, g.num_cols, g.malformed),
            (0, 1, 0, false)
        );
    }

    fn charset() -> Vec<String> {
        ["sos", "<tr>", TD, "</tr>", "eos"]
            .iter()
            .map(|s| s.to_string())
            .collect()
    }
    fn onehot_array(idx: &[usize]) -> ndarray::Array2<f32> {
        let rows: Vec<f32> = idx
            .iter()
            .flat_map(|&i| {
                let mut r = [0.0f32; 5];
                r[i] = 1.0;
                r
            })
            .collect();
        ndarray::Array2::from_shape_vec((idx.len(), 5), rows).unwrap()
    }

    #[test]
    fn argmax_skips_sos_and_stops_at_eos() {
        let cs = charset();
        let a = onehot_array(&[4, 0, 1, 2, 3, 4, 1]);
        let t = argmax_tokens(a.view(), &cs, 500);
        assert_eq!(
            t.iter().map(|s| s.token.as_str()).collect::<Vec<_>>(),
            vec!["<tr>", TD, "</tr>"]
        );
        assert_eq!(t.iter().map(|s| s.step).collect::<Vec<_>>(), vec![2, 3, 4]);
        assert_eq!(argmax_tokens(a.view(), &cs, 3).len(), 1);
    }

    #[test]
    fn argmax_eos_first_yields_empty_and_decodes_to_empty_grid() {
        let cs = charset();
        let a = onehot_array(&[4, 1, 2]);
        let t = argmax_tokens(a.view(), &cs, 500);
        assert_eq!(
            t.iter().map(|s| s.token.as_str()).collect::<Vec<_>>(),
            vec!["<tr>", TD]
        );
        let a = onehot_array(&[4, 4]);
        assert!(argmax_tokens(a.view(), &cs, 500).is_empty());
        assert_eq!(decode_structure_tokens(&[]).num_cols, 0);
    }

    #[test]
    fn argmax_truncated_at_max_tokens_and_logit_rows() {
        let cs = charset();
        let a = onehot_array(&[0, 1, 2, 1, 2, 3]);
        let t = argmax_tokens(a.view(), &cs, 4);
        assert_eq!(t.len(), 3);
        // Raw logits (not a distribution) go through softmax; one-hot rows pass through.
        assert!((t[0].prob - 1.0).abs() < 1e-6);
        let logits = ndarray::arr2(&[[0.0f32, 5.0, 0.0, 0.0, 0.0], [0.0, 0.0, 9.0, 0.0, 0.0]]);
        let t = argmax_tokens(logits.view(), &cs, 500);
        assert_eq!(t.len(), 2);
        assert!(t[0].prob > 0.9 && t[0].prob < 1.0);
    }
}
