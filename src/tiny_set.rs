/// A trait for tiny enums with ≤8 variants where the discriminant of each
/// variant is <8.
pub trait TinyEnum: Sized + Copy + 'static {
    /// A list of all variants of the enum.
    const VARIANTS: &'static [Self];

    /// The bit mask of the given variant.
    fn bit_mask(self) -> u8;
}

/// A set of tiny enums implemented as a bit set.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct TinySet<T> {
    data: u8,
    _phantom: std::marker::PhantomData<T>,
}

impl<T> TinySet<T> {
    /// Creates a new empty set.
    pub const fn new() -> Self {
        Self {
            data: 0,
            _phantom: std::marker::PhantomData,
        }
    }

    pub(crate) const fn from_raw_unchecked(data: u8) -> Self {
        Self {
            data,
            _phantom: std::marker::PhantomData,
        }
    }

    pub const fn is_empty(self) -> bool {
        self.data == 0
    }
    pub const fn len(self) -> usize {
        self.data.count_ones() as usize
    }

    pub const fn is_disjoint(self, other: Self) -> bool {
        (self.data & other.data) == 0
    }
    pub const fn is_subset(self, other: Self) -> bool {
        (self.data & other.data) == self.data
    }
    pub const fn is_superset(self, other: Self) -> bool {
        (self.data & other.data) == other.data
    }

    pub const fn union(self, other: Self) -> Self {
        Self {
            data: self.data | other.data,
            _phantom: std::marker::PhantomData,
        }
    }
    pub const fn intersection(self, other: Self) -> Self {
        Self {
            data: self.data & other.data,
            _phantom: std::marker::PhantomData,
        }
    }
    pub const fn difference(self, other: Self) -> Self {
        Self {
            data: self.data & !other.data,
            _phantom: std::marker::PhantomData,
        }
    }
    pub const fn symmetric_difference(self, other: Self) -> Self {
        Self {
            data: self.data ^ other.data,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<T> TinySet<T>
where
    T: TinyEnum,
{
    pub fn contains(self, value: T) -> bool {
        let mask = value.bit_mask();
        (self.data & mask) != 0
    }
    pub fn insert(&mut self, value: T) -> bool {
        let mask = value.bit_mask();
        let old = self.data;
        self.data |= mask;
        old != self.data
    }
    pub fn remove(&mut self, value: T) -> bool {
        let mask = value.bit_mask();
        let old = self.data;
        self.data &= !mask;
        old != self.data
    }
}

impl<T: TinyEnum> IntoIterator for TinySet<T> {
    type Item = T;
    type IntoIter = TinySetIter<T>;

    fn into_iter(self) -> Self::IntoIter {
        TinySetIter {
            data: self,
            index: 0,
            _phantom: std::marker::PhantomData,
        }
    }
}

pub struct TinySetIter<T> {
    data: TinySet<T>,
    index: u8,
    _phantom: std::marker::PhantomData<T>,
}
impl<T: TinyEnum> Iterator for TinySetIter<T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        if !self.data.is_empty() {
            while (self.index as usize) < T::VARIANTS.len() {
                let variant = T::VARIANTS[self.index as usize];
                self.index += 1;

                if self.data.remove(variant) {
                    return Some(variant);
                }
            }
        }

        None
    }
}
