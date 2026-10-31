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

use crate::orient::probabilities;
use crate::table::{BBox, TableCell, TableRow, TableStructure};
use crate::TextResult;
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

/// Convert a location-head quad (4 corner points, `x0 y0 x1 y1 x2 y2 x3 y3`)
/// to a box in page pixels.
///
/// The model sees the crop resized to fit a square canvas, so the quad values
/// are fractions of that canvas and `crop pixels = q * max(crop_w, crop_h)`
/// (the same mapping as `bobine` `rapid_table.rs`). x is clamped to
/// `[0, crop_w]`, y to `[0, crop_h]`, the box is the min/max of the four
/// points truncated to integers, and `origin` (the crop's top-left on the
/// page) is added. Returns `None` for a quad shorter than 8 values or one
/// whose values are all below `1e-6` (a cell the model gave no location).
pub(crate) fn quad_to_bbox(quad: &[f32], crop_w: u32, crop_h: u32, origin: (i32, i32)) -> Option<BBox> {
    if quad.len() < 8 || quad[..8].iter().all(|&v| v < 1e-6) {
        return None;
    }
    let scale = crop_w.max(crop_h) as f32;
    let (w, h) = (crop_w as f32, crop_h as f32);
    let (mut x_min, mut x_max) = (f32::INFINITY, f32::NEG_INFINITY);
    let (mut y_min, mut y_max) = (f32::INFINITY, f32::NEG_INFINITY);
    for p in quad[..8].chunks_exact(2) {
        let x = (p[0] * scale).clamp(0.0, w);
        let y = (p[1] * scale).clamp(0.0, h);
        x_min = x_min.min(x);
        x_max = x_max.max(x);
        y_min = y_min.min(y);
        y_max = y_max.max(y);
    }
    Some(BBox {
        x_min: x_min as i32 + origin.0,
        y_min: y_min as i32 + origin.1,
        x_max: x_max as i32 + origin.0,
        y_max: y_max as i32 + origin.1,
    })
}

/// Build the table structure from the decoded grid. `tokens` is the slice the
/// grid was decoded from (`GridCell::token_index` indexes it, and
/// `tokens[i].step` is the row of `loc` holding that token's quad). A cell
/// whose step is beyond `loc` or whose quad is empty gets a degenerate box at
/// `origin`, which [`assign_lines_to_cells`] never fills. Cell texts are empty.
pub(crate) fn build_structure(
    grid: &DecodedGrid,
    tokens: &[StepToken],
    loc: ArrayView2<f32>,
    crop_w: u32,
    crop_h: u32,
    origin: (i32, i32),
) -> TableStructure {
    let degenerate = BBox { x_min: origin.0, y_min: origin.1, x_max: origin.0, y_max: origin.1 };
    let mut rows: Vec<Vec<TableCell>> = vec![Vec::new(); grid.num_rows];
    for g in &grid.cells {
        let token = tokens.get(g.token_index);
        let bbox = token
            .filter(|t| t.step < loc.nrows())
            .and_then(|t| quad_to_bbox(&loc.row(t.step).to_vec(), crop_w, crop_h, origin))
            .unwrap_or(degenerate);
        let cell = TableCell {
            bbox,
            row: g.row,
            col: g.col,
            row_span: g.row_span,
            col_span: g.col_span,
            text: String::new(),
            score: token.map_or(0.0, |t| t.prob),
        };
        if g.row >= rows.len() {
            rows.resize(g.row + 1, Vec::new());
        }
        rows[g.row].push(cell);
    }
    for r in &mut rows {
        r.sort_by_key(|c| c.col);
    }
    TableStructure {
        num_rows: grid.num_rows,
        num_cols: grid.num_cols,
        rows: rows.into_iter().map(|cells| TableRow { cells }).collect(),
        unassigned_lines: Vec::new(),
    }
}

fn box_area(b: &BBox) -> i64 {
    (b.x_max - b.x_min).max(0) as i64 * (b.y_max - b.y_min).max(0) as i64
}

/// Intersection over union of a line frame and a cell box.
fn frame_iou(line: &TextResult, b: &BBox) -> f32 {
    let f = &line.frame;
    let (fx0, fy0, fx1, fy1) = (f.left, f.top, f.left + f.width, f.top + f.height);
    let iw = (fx1.min(b.x_max as f32) - fx0.max(b.x_min as f32)).max(0.0);
    let ih = (fy1.min(b.y_max as f32) - fy0.max(b.y_min as f32)).max(0.0);
    let inter = iw * ih;
    let union = f.width.max(0.0) * f.height.max(0.0) + box_area(b) as f32 - inter;
    if inter > 0.0 && union > 0.0 {
        inter / union
    } else {
        0.0
    }
}

/// Index of the cell (in row-major order) that receives `line`, if any.
/// `boxes` holds the cell boxes in row-major order; zero-area boxes never match.
fn target_cell(line: &TextResult, boxes: &[BBox]) -> Option<usize> {
    let f = &line.frame;
    let (cx, cy) = (f.left + f.width / 2.0, f.top + f.height / 2.0);
    let mut best: Option<(usize, i64)> = None;
    for (i, b) in boxes.iter().enumerate() {
        let inside = cx >= b.x_min as f32 && cx <= b.x_max as f32 && cy >= b.y_min as f32 && cy <= b.y_max as f32;
        let area = box_area(b);
        if inside && area > 0 && best.is_none_or(|(_, a)| area < a) {
            best = Some((i, area));
        }
    }
    if let Some((i, _)) = best {
        return Some(i);
    }
    let mut best: Option<(usize, f32)> = None;
    for (i, b) in boxes.iter().enumerate() {
        let iou = frame_iou(line, b);
        if iou > 0.0 && best.is_none_or(|(_, v)| iou > v) {
            best = Some((i, iou));
        }
    }
    best.map(|(i, _)| i)
}

/// Order text lines for reading (used for table cells and for whole text
/// regions): lines whose vertical centres differ by at most half the smaller
/// line height form a band, bands run top to bottom and each band left to right.
pub(crate) fn reading_order(lines: &mut Vec<TextResult>) {
    let centre_y = |l: &TextResult| l.frame.top + l.frame.height / 2.0;
    lines.sort_by(|a, b| centre_y(a).total_cmp(&centre_y(b)));
    let mut bands: Vec<Vec<TextResult>> = Vec::new();
    let mut anchor: Option<(f32, f32)> = None; // (centre y, height) of the band's first line
    for l in lines.drain(..) {
        let same_band = anchor.is_some_and(|(cy, h)| (centre_y(&l) - cy).abs() <= 0.5 * h.min(l.frame.height));
        if same_band {
            bands.last_mut().expect("a band exists once an anchor is set").push(l);
        } else {
            anchor = Some((centre_y(&l), l.frame.height));
            bands.push(vec![l]);
        }
    }
    for mut band in bands {
        band.sort_by(|a, b| a.frame.left.total_cmp(&b.frame.left));
        lines.extend(band);
    }
}

/// Assign OCR lines (in the same page coordinates as the cell boxes) to cells
/// and fill `cell.text`. A line goes to the smallest cell box containing the
/// line centre (inclusive); otherwise to the cell with the highest IoU > 0
/// (ties: first in row-major order); otherwise it is appended to
/// `unassigned_lines`. Degenerate cell boxes never match. The lines of a cell
/// are put in reading order, trimmed, and joined with single spaces; a cell
/// that receives no line keeps its text.
pub(crate) fn assign_lines_to_cells(structure: &mut TableStructure, lines: Vec<TextResult>) {
    let boxes: Vec<BBox> = structure.rows.iter().flat_map(|r| r.cells.iter().map(|c| c.bbox)).collect();
    let mut per_cell: Vec<Vec<TextResult>> = vec![Vec::new(); boxes.len()];
    for line in lines {
        match target_cell(&line, &boxes) {
            Some(i) => per_cell[i].push(line),
            None => structure.unassigned_lines.push(line),
        }
    }
    let cells = structure.rows.iter_mut().flat_map(|r| r.cells.iter_mut());
    for (cell, mut assigned) in cells.zip(per_cell) {
        if assigned.is_empty() {
            continue;
        }
        reading_order(&mut assigned);
        cell.text = assigned
            .iter()
            .map(|l| l.text.trim())
            .filter(|t| !t.is_empty())
            .collect::<Vec<_>>()
            .join(" ");
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

    use crate::table::{BBox, TableCell, TableRow, TableStructure};
    use crate::{Frame, TextResult};

    fn line(text: &str, l: f32, t: f32, w: f32, h: f32) -> TextResult {
        TextResult {
            text: text.into(),
            score: 0.9,
            box_points: [(l, t), (l + w, t), (l + w, t + h), (l, t + h)],
            frame: Frame::new(w, h, t, l),
        }
    }
    fn cell(r: usize, c: usize, b: [i32; 4], cs: usize, text: &str) -> TableCell {
        TableCell {
            bbox: BBox { x_min: b[0], y_min: b[1], x_max: b[2], y_max: b[3] },
            row: r,
            col: c,
            row_span: 1,
            col_span: cs,
            text: text.into(),
            score: 1.0,
        }
    }
    fn table_2x2() -> TableStructure {
        TableStructure {
            num_rows: 2,
            num_cols: 2,
            unassigned_lines: vec![],
            rows: vec![
                TableRow { cells: vec![cell(0, 0, [0, 0, 100, 50], 1, ""), cell(0, 1, [100, 0, 200, 50], 1, "")] },
                TableRow { cells: vec![cell(1, 0, [0, 50, 100, 100], 1, ""), cell(1, 1, [100, 50, 200, 100], 1, "")] },
            ],
        }
    }

    #[test]
    fn quad_scales_by_long_side_and_offsets() {
        let b = quad_to_bbox(&[0.1, 0.1, 0.5, 0.1, 0.5, 0.2, 0.1, 0.2], 976, 488, (300, 700)).unwrap();
        assert_eq!((b.x_min, b.y_min, b.x_max, b.y_max), (300 + 97, 700 + 97, 300 + 488, 700 + 195));
        assert!(quad_to_bbox(&[0.0; 8], 976, 488, (0, 0)).is_none());
    }

    #[test]
    fn quad_clamps_to_the_crop() {
        // 1.5 * 100 = 150 clamps to the crop width, y 0.9 * 100 = 90 clamps to the crop height 50.
        let b = quad_to_bbox(&[0.2, 0.9, 1.5, 0.9, 1.5, 0.95, 0.2, 0.95], 100, 50, (0, 0)).unwrap();
        assert_eq!((b.x_min, b.y_min, b.x_max, b.y_max), (20, 50, 100, 50));
        assert!(quad_to_bbox(&[0.1, 0.1, 0.2], 100, 50, (0, 0)).is_none());
    }

    #[test]
    fn assign_by_centre_then_iou_then_unassigned() {
        let mut s = table_2x2();
        assign_lines_to_cells(
            &mut s,
            vec![
                line("Цена", 120.0, 10.0, 60.0, 20.0),   // centre in (0,1)
                line("12", 90.0, 60.0, 30.0, 20.0),      // centre (105,70) in (1,1)
                line("край", 195.0, 95.0, 30.0, 20.0),   // centre (210,105) outside; IoU > 0 only with (1,1)
                line("извън", 400.0, 400.0, 30.0, 20.0), // no overlap
            ],
        );
        assert_eq!(s.rows[0].cells[1].text, "Цена");
        assert_eq!(s.rows[1].cells[1].text, "12 край");
        assert_eq!(s.unassigned_lines.iter().map(|l| l.text.as_str()).collect::<Vec<_>>(), vec!["извън"]);
    }

    #[test]
    fn iou_tie_goes_to_first_cell_in_row_major_order() {
        let mut s = table_2x2();
        // Centre (100, 120) is outside every cell (below the table); the frame
        // overlaps (1,0) and (1,1) by the same area.
        assign_lines_to_cells(&mut s, vec![line("т", 90.0, 90.0, 20.0, 60.0)]);
        assert_eq!((s.rows[1].cells[0].text.as_str(), s.rows[1].cells[1].text.as_str()), ("т", ""));
        assert!(s.unassigned_lines.is_empty());
    }

    #[test]
    fn lines_in_one_cell_join_in_reading_order() {
        let mut s = table_2x2();
        assign_lines_to_cells(
            &mut s,
            vec![line("б", 50.0, 5.0, 20.0, 10.0), line("в", 5.0, 30.0, 20.0, 10.0), line("а", 5.0, 6.0, 20.0, 10.0)],
        );
        assert_eq!(s.rows[0].cells[0].text, "а б в");
    }

    #[test]
    fn overlapping_cells_pick_smallest_containing() {
        let mut s = table_2x2();
        s.rows[0].cells.insert(0, cell(0, 0, [0, 0, 200, 100], 2, ""));
        assign_lines_to_cells(&mut s, vec![line("а", 10.0, 10.0, 20.0, 10.0)]);
        assert_eq!((s.rows[0].cells[0].text.as_str(), s.rows[0].cells[1].text.as_str()), ("", "а"));
    }

    #[test]
    fn degenerate_cells_never_receive_lines() {
        let mut s = table_2x2();
        s.rows[0].cells[0].bbox = BBox { x_min: 30, y_min: 20, x_max: 30, y_max: 20 };
        assign_lines_to_cells(&mut s, vec![line("а", 20.0, 15.0, 20.0, 10.0)]); // centre (30, 20)
        assert_eq!(s.rows[0].cells[0].text, "");
        assert_eq!(s.unassigned_lines.len(), 1);
    }

    #[test]
    fn assign_with_zero_cells_leaves_all_unassigned() {
        let mut s = TableStructure { rows: vec![], num_rows: 0, num_cols: 0, unassigned_lines: vec![] };
        assign_lines_to_cells(&mut s, vec![line("а", 0.0, 0.0, 10.0, 10.0), line("б", 20.0, 0.0, 10.0, 10.0)]);
        assert_eq!(s.unassigned_lines.len(), 2);
    }

    #[test]
    fn build_structure_degenerate_box_for_missing_quad() {
        let grid = decode_structure_tokens(&["<tr>", "<td></td>", "</tr>"]);
        let toks = vec![
            StepToken { step: 1, token: "<tr>".into(), prob: 0.9 },
            StepToken { step: 2, token: "<td></td>".into(), prob: 0.8 },
            StepToken { step: 3, token: "</tr>".into(), prob: 0.9 },
        ];
        let s = build_structure(&grid, &toks, ndarray::Array2::<f32>::zeros((5, 8)).view(), 100, 50, (300, 700));
        let c = &s.rows[0].cells[0];
        assert_eq!((c.bbox.x_min, c.bbox.y_min, c.bbox.x_max, c.bbox.y_max, c.score), (300, 700, 300, 700, 0.8));
    }

    #[test]
    fn build_structure_uses_step_row_origin_and_score() {
        // Tokens: <tr> <td></td> <td></td> </tr> at steps 1..=4; cells take the
        // location rows of their own steps (2 and 3), not their token index.
        let tokens = ["<tr>", "<td></td>", "<td></td>", "</tr>"];
        let grid = decode_structure_tokens(&tokens);
        let toks: Vec<StepToken> = tokens
            .iter()
            .enumerate()
            .map(|(i, t)| StepToken { step: i + 1, token: (*t).into(), prob: 0.5 + i as f32 / 10.0 })
            .collect();
        let mut loc = ndarray::Array2::<f32>::zeros((6, 8));
        loc.row_mut(2).assign(&ndarray::arr1(&[0.0, 0.0, 0.25, 0.0, 0.25, 0.5, 0.0, 0.5]));
        loc.row_mut(3).assign(&ndarray::arr1(&[0.25, 0.0, 0.5, 0.0, 0.5, 0.5, 0.25, 0.5]));
        let s = build_structure(&grid, &toks, loc.view(), 200, 100, (10, 20));
        assert_eq!((s.num_rows, s.num_cols), (1, 2));
        let b0 = s.rows[0].cells[0].bbox;
        let b1 = s.rows[0].cells[1].bbox;
        assert_eq!((b0.x_min, b0.y_min, b0.x_max, b0.y_max), (10, 20, 10 + 50, 20 + 100));
        assert_eq!((b1.x_min, b1.y_min, b1.x_max, b1.y_max), (10 + 50, 20, 10 + 100, 20 + 100));
        assert_eq!((s.rows[0].cells[0].score, s.rows[0].cells[1].score), (0.6, 0.7));
    }

    #[test]
    fn build_structure_step_beyond_location_rows_is_degenerate() {
        let grid = decode_structure_tokens(&["<tr>", "<td></td>", "</tr>"]);
        let toks = vec![
            StepToken { step: 1, token: "<tr>".into(), prob: 0.9 },
            StepToken { step: 40, token: "<td></td>".into(), prob: 0.8 },
            StepToken { step: 41, token: "</tr>".into(), prob: 0.9 },
        ];
        let loc = ndarray::Array2::<f32>::from_elem((5, 8), 0.5);
        let s = build_structure(&grid, &toks, loc.view(), 100, 50, (7, 9));
        let b = s.rows[0].cells[0].bbox;
        assert_eq!((b.x_min, b.y_min, b.x_max, b.y_max), (7, 9, 7, 9));
    }

    #[test]
    fn build_structure_zero_cells_is_empty() {
        let grid = decode_structure_tokens(&[]);
        let s = build_structure(&grid, &[], ndarray::Array2::<f32>::zeros((0, 8)).view(), 100, 50, (0, 0));
        assert_eq!((s.num_rows, s.num_cols, s.rows.len()), (0, 0, 0));
    }
}
