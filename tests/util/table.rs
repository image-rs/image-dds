pub struct PrettyTable {
    cells: Vec<String>,
    width: usize,
    height: usize,
}
impl PrettyTable {
    pub fn new_empty(width: usize, height: usize) -> Self {
        Self {
            cells: vec![String::new(); width * height],
            width,
            height,
        }
    }
    pub fn from_header<S: AsRef<str>>(header: &[S]) -> Self {
        let mut table = Self::new_empty(header.len(), 0);
        table.add_row(header);
        table
    }
    pub fn height(&self) -> usize {
        self.height
    }

    pub fn get(&self, x: usize, y: usize) -> &str {
        &self.cells[y * self.width + x]
    }
    pub fn get_mut(&mut self, x: usize, y: usize) -> &mut String {
        &mut self.cells[y * self.width + x]
    }

    #[allow(unused)]
    pub fn set(&mut self, x: usize, y: usize, value: impl Into<String>) {
        *self.get_mut(x, y) = value.into();
    }

    #[track_caller]
    pub fn add_row<S: AsRef<str>>(&mut self, row: &[S]) {
        assert!(row.len() == self.width);
        self.height += 1;
        for cell in row {
            self.cells.push(cell.as_ref().to_string());
        }
    }
    pub fn add_empty_row(&mut self) {
        self.height += 1;
        for _ in 0..self.width {
            self.cells.push(String::new());
        }
    }

    pub fn print(&self, out: &mut String) {
        let column_width: Vec<usize> = (0..self.width)
            .map(|x| {
                (0..self.height)
                    .map(|y| self.get(x, y).chars().count())
                    .max()
                    .unwrap()
            })
            .collect();

        let mut line = String::new();
        for y in 0..self.height {
            #[allow(clippy::needless_range_loop)]
            for x in 0..self.width {
                let cell = self.get(x, y);
                line.push_str(cell);
                for _ in 0..column_width[x] - cell.chars().count() {
                    line.push(' ');
                }
                line.push_str("  ");
            }
            out.push_str(line.trim_end());
            out.push('\n');
            line.clear();
        }
    }

    pub fn print_markdown(&self, out: &mut String) {
        let column_width: Vec<usize> = (0..self.width)
            .map(|x| {
                (0..self.height)
                    .map(|y| self.get(x, y).chars().count())
                    .max()
                    .unwrap()
            })
            .collect();

        for y in 0..self.height {
            #[allow(clippy::needless_range_loop)]
            for x in 0..self.width {
                let cell = self.get(x, y);
                out.push_str("| ");
                out.push_str(cell);
                for _ in 0..column_width[x] - cell.chars().count() {
                    out.push(' ');
                }
                out.push(' ');
            }

            // poor man's trim
            while let Some(last) = out.chars().last() {
                if last == ' ' {
                    out.pop();
                } else {
                    break;
                }
            }

            out.push('\n');

            if y == 0 {
                #[allow(clippy::needless_range_loop)]
                for x in 0..self.width {
                    out.push_str("| ");
                    for _ in 0..column_width[x] {
                        out.push('-');
                    }
                    if x != self.width - 1 {
                        out.push(' ');
                    }
                }
                out.push('\n');
            }
        }
    }
}
impl std::fmt::Display for PrettyTable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut out = String::new();
        self.print(&mut out);
        write!(f, "{out}")
    }
}
