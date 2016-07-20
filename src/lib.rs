// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Copyright 2016 The bit-set-array developers.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! An implementation of a set using a bit array as an underlying
//! representation for holding unsigned numerical elements.
//!
//! It should also be noted that the amount of storage necessary for holding a
//! set of objects is proportional to the maximum of the objects when viewed
//! as a `usize`.
//!
//! # Examples
//!
//! ```
//! extern crate typenum;
//! # extern crate bitarray_set;
//! use typenum::{Unsigned, U8};
//! use bitarray_set::BitArraySet;
//!
//! # fn main() {
//! // It's a regular set
//! let mut s = BitArraySet::<u32, U8>::new();
//! s.insert(0);
//! s.insert(3);
//! s.insert(7);
//!
//! s.remove(7);
//!
//! if !s.contains(7) {
//!     println!("There is no 7");
//! }
//!
//! // Can initialize from a `BitArray`
//! let other = BitArraySet::<u32, U8>::from_bytes(&[0b11010000]);
//!
//! s.union_with(&other);
//!
//! // Print 0, 1, 3 in some order
//! for x in s.iter() {
//!     println!("{}", x);
//! }
//!
//! // Can convert back to a `BitArray`
//! let bv = s.into_bit_array();
//! assert!(bv[3]);
//! # }
//! ```

#![cfg_attr(all(test, feature = "nightly"), feature(test))]
#[cfg(all(test, feature = "nightly"))] extern crate test;
#[cfg(all(test, feature = "nightly"))] extern crate rand;
extern crate bit_vec;
extern crate bit_array;
extern crate typenum;
extern crate generic_array;

use bit_vec::BitBlock;
use bit_array::{BitsIn, BitArray, Blocks};
use std::cmp::Ordering;
use std::cmp;
use std::fmt;
use std::hash;
use std::ops::*;
use std::iter::{self, Chain, Enumerate, FromIterator, Repeat, Skip, Take};
use typenum::{Unsigned, NonZero};

type MatchWords<'a, B> = Chain<Enumerate<Blocks<'a, B>>, Skip<Take<Enumerate<Repeat<B>>>>>;

// Take two BitArray's, and return iterators of their words, where the shorter one
// has been padded with 0's
fn match_words<'a, 'b, B: BitsIn + BitBlock + Default, NBits: Unsigned + NonZero>(a: &'a BitArray<B, NBits>, b: &'b BitArray<B, NBits>)
    -> (MatchWords<'a, B>, MatchWords<'b, B>)
    where NBits: Add<<B as BitsIn>::Output>,
    <NBits as Add<<B as BitsIn>::Output>>::Output: Sub<typenum::B1>,
    <<NBits as Add<<B as BitsIn>::Output>>::Output as Sub<typenum::B1>>::Output: Div<<B as BitsIn>::Output>,
    <<<NBits as Add<<B as BitsIn>::Output>>::Output as Sub<typenum::B1>>::Output as Div<<B as BitsIn>::Output>>::Output: generic_array::ArrayLength<B>
{
    let a_len = a.len();
    let b_len = b.len();

    // have to uselessly pretend to pad the longer one for type matching
    if a_len < b_len {
        (a.blocks().enumerate().chain(iter::repeat(B::zero()).enumerate().take(b_len).skip(a_len)),
         b.blocks().enumerate().chain(iter::repeat(B::zero()).enumerate().take(0).skip(0)))
    } else {
        (a.blocks().enumerate().chain(iter::repeat(B::zero()).enumerate().take(0).skip(0)),
         b.blocks().enumerate().chain(iter::repeat(B::zero()).enumerate().take(a_len).skip(b_len)))
    }
}

pub struct BitArraySet<B: BitsIn, NBits: Unsigned + NonZero> 
    where NBits: Add<<B as BitsIn>::Output>,
    <NBits as Add<<B as BitsIn>::Output>>::Output: Sub<typenum::B1>,
    <<NBits as Add<<B as BitsIn>::Output>>::Output as Sub<typenum::B1>>::Output: Div<<B as BitsIn>::Output>,
    <<<NBits as Add<<B as BitsIn>::Output>>::Output as Sub<typenum::B1>>::Output as Div<<B as BitsIn>::Output>>::Output: generic_array::ArrayLength<B>
{
    bit_array: BitArray<B, NBits>,
}

impl<B: BitsIn + BitBlock + Default, NBits: Unsigned + NonZero>  Clone for BitArraySet<B, NBits> 
    where NBits: Add<<B as BitsIn>::Output>,
    <NBits as Add<<B as BitsIn>::Output>>::Output: Sub<typenum::B1>,
    <<NBits as Add<<B as BitsIn>::Output>>::Output as Sub<typenum::B1>>::Output: Div<<B as BitsIn>::Output>,
    <<<NBits as Add<<B as BitsIn>::Output>>::Output as Sub<typenum::B1>>::Output as Div<<B as BitsIn>::Output>>::Output: generic_array::ArrayLength<B>
{
    fn clone(&self) -> Self {
        BitArraySet {
            bit_array: self.bit_array.clone(),
        }
    }

    fn clone_from(&mut self, other: &Self) {
        self.bit_array.clone_from(&other.bit_array);
    }
}

impl<B: BitsIn + BitBlock + Default, NBits: Unsigned + NonZero> Default for BitArraySet<B, NBits> 
    where NBits: Add<<B as BitsIn>::Output>,
    <NBits as Add<<B as BitsIn>::Output>>::Output: Sub<typenum::B1>,
    <<NBits as Add<<B as BitsIn>::Output>>::Output as Sub<typenum::B1>>::Output: Div<<B as BitsIn>::Output>,
    <<<NBits as Add<<B as BitsIn>::Output>>::Output as Sub<typenum::B1>>::Output as Div<<B as BitsIn>::Output>>::Output: generic_array::ArrayLength<B>
 {
    #[inline]
    fn default() -> Self { BitArraySet { bit_array: Default::default() } }
}

impl<B: BitsIn + BitBlock + Default, NBits: Unsigned + NonZero> FromIterator<usize> for BitArraySet<B, NBits> 
    where NBits: Add<<B as BitsIn>::Output>,
    <NBits as Add<<B as BitsIn>::Output>>::Output: Sub<typenum::B1>,
    <<NBits as Add<<B as BitsIn>::Output>>::Output as Sub<typenum::B1>>::Output: Div<<B as BitsIn>::Output>,
    <<<NBits as Add<<B as BitsIn>::Output>>::Output as Sub<typenum::B1>>::Output as Div<<B as BitsIn>::Output>>::Output: generic_array::ArrayLength<B>
{
    fn from_iter<I: IntoIterator<Item = usize>>(iter: I) -> Self {
        let mut ret = Self::default();
        ret.extend(iter);
        ret
    }
}

impl<B: BitsIn + BitBlock + Default + BitAnd + BitOr, NBits: Unsigned + NonZero> Extend<usize> for BitArraySet<B, NBits> 
    where NBits: Add<<B as BitsIn>::Output>,
    <NBits as Add<<B as BitsIn>::Output>>::Output: Sub<typenum::B1>,
    <<NBits as Add<<B as BitsIn>::Output>>::Output as Sub<typenum::B1>>::Output: Div<<B as BitsIn>::Output>,
    <<<NBits as Add<<B as BitsIn>::Output>>::Output as Sub<typenum::B1>>::Output as Div<<B as BitsIn>::Output>>::Output: generic_array::ArrayLength<B>
{
    #[inline]
    fn extend<I: IntoIterator<Item = usize>>(&mut self, iter: I) {
        for i in iter {
            self.insert(i);
        }
    }
}

impl<B: BitsIn + BitBlock + Default, NBits: Unsigned + NonZero> PartialOrd for BitArraySet<B, NBits> 
    where NBits: Add<<B as BitsIn>::Output>,
    <NBits as Add<<B as BitsIn>::Output>>::Output: Sub<typenum::B1>,
    <<NBits as Add<<B as BitsIn>::Output>>::Output as Sub<typenum::B1>>::Output: Div<<B as BitsIn>::Output>,
    <<<NBits as Add<<B as BitsIn>::Output>>::Output as Sub<typenum::B1>>::Output as Div<<B as BitsIn>::Output>>::Output: generic_array::ArrayLength<B>
{
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.iter().partial_cmp(other)
    }
}

impl<B: BitsIn + BitBlock + Default, NBits: Unsigned + NonZero> Ord for BitArraySet<B, NBits> 
    where NBits: Add<<B as BitsIn>::Output>,
    <NBits as Add<<B as BitsIn>::Output>>::Output: Sub<typenum::B1>,
    <<NBits as Add<<B as BitsIn>::Output>>::Output as Sub<typenum::B1>>::Output: Div<<B as BitsIn>::Output>,
    <<<NBits as Add<<B as BitsIn>::Output>>::Output as Sub<typenum::B1>>::Output as Div<<B as BitsIn>::Output>>::Output: generic_array::ArrayLength<B>
{
    #[inline]
    fn cmp(&self, other: &Self) -> Ordering {
        self.iter().cmp(other)
    }
}

impl<B: BitsIn + BitBlock + Default, NBits: Unsigned + NonZero> PartialEq for BitArraySet<B, NBits> 
    where NBits: Add<<B as BitsIn>::Output>,
    <NBits as Add<<B as BitsIn>::Output>>::Output: Sub<typenum::B1>,
    <<NBits as Add<<B as BitsIn>::Output>>::Output as Sub<typenum::B1>>::Output: Div<<B as BitsIn>::Output>,
    <<<NBits as Add<<B as BitsIn>::Output>>::Output as Sub<typenum::B1>>::Output as Div<<B as BitsIn>::Output>>::Output: generic_array::ArrayLength<B>
{
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.iter().eq(other)
    }
}

impl<B: BitsIn + BitBlock + Default, NBits: Unsigned + NonZero> Eq for BitArraySet<B, NBits> 
    where NBits: Add<<B as BitsIn>::Output>,
    <NBits as Add<<B as BitsIn>::Output>>::Output: Sub<typenum::B1>,
    <<NBits as Add<<B as BitsIn>::Output>>::Output as Sub<typenum::B1>>::Output: Div<<B as BitsIn>::Output>,
    <<<NBits as Add<<B as BitsIn>::Output>>::Output as Sub<typenum::B1>>::Output as Div<<B as BitsIn>::Output>>::Output: generic_array::ArrayLength<B>
{}

impl<B: BitsIn + BitBlock + Default, NBits: Unsigned + NonZero> BitArraySet<B, NBits> 
    where NBits: Add<<B as BitsIn>::Output>,
    <NBits as Add<<B as BitsIn>::Output>>::Output: Sub<typenum::B1>,
    <<NBits as Add<<B as BitsIn>::Output>>::Output as Sub<typenum::B1>>::Output: Div<<B as BitsIn>::Output>,
    <<<NBits as Add<<B as BitsIn>::Output>>::Output as Sub<typenum::B1>>::Output as Div<<B as BitsIn>::Output>>::Output: generic_array::ArrayLength<B>
{

    /// Creates a new empty `BitArraySet`.
    ///
    /// # Examples
    ///
    /// ```
    /// extern crate typenum;
    /// # extern crate bitarray_set;
    /// use typenum::{Unsigned, U8};
    /// use bitarray_set::BitArraySet;
    ///
    /// # fn main() {
    /// let mut s = BitArraySet::<u32, U8>::new();
    /// # }
    /// ```
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates a new `BitArraySet` from the given bit array.
    ///
    /// # Examples
    ///
    /// ```
    /// extern crate typenum;
    /// extern crate bit_array;
    /// extern crate bitarray_set;
    /// use typenum::{Unsigned, U8};
    /// use bit_array::BitArray;
    /// use bitarray_set::BitArraySet;
    ///
    /// fn main() {
    ///     let bv = BitArray::<u32, U8>::from_bytes(&[0b01100000]);
    ///     let s = BitArraySet::from_bit_array(bv);
    ///
    ///     // Print 1, 2 in arbitrary order
    ///     for x in s.iter() {
    ///         println!("{}", x);
    ///     }
    /// }
    /// ```
    #[inline]
    pub fn from_bit_array(bit_array: BitArray<B, NBits>) -> Self {
        BitArraySet { bit_array: bit_array }
    }

    pub fn from_bytes(bytes: &[u8]) -> Self {
        BitArraySet { bit_array: BitArray::from_bytes(bytes) }
    }

    /// Consumes this set to return the underlying bit array.
    ///
    /// # Examples
    ///
    /// ```
    /// extern crate typenum;
    /// # extern crate bitarray_set;
    /// use typenum::{Unsigned, U8};
    /// use bitarray_set::BitArraySet;
    ///
    /// # fn main() {
    /// let mut s = BitArraySet::<u32, U8>::new();
    /// s.insert(0);
    /// s.insert(3);
    ///
    /// let bv = s.into_bit_array();
    /// assert!(bv[0]);
    /// assert!(bv[3]);
    /// # }
    /// ```
    #[inline]
    pub fn into_bit_array(self) -> BitArray<B, NBits> {
        self.bit_array
    }

    /// Returns a reference to the underlying bit array.
    ///
    /// # Examples
    ///
    /// ```
    /// extern crate typenum;
    /// # extern crate bitarray_set;
    /// use typenum::{Unsigned, U8};
    /// use bitarray_set::BitArraySet;
    ///
    /// # fn main() {
    /// let mut s = BitArraySet::<u32, U8>::new();
    /// s.insert(0);
    ///
    /// let bv = s.get_ref();
    /// assert_eq!(bv[0], true);
    /// # }
    /// ```
    #[inline]
    pub fn get_ref(&self) -> &BitArray<B, NBits> {
        &self.bit_array
    }

    #[inline]
    fn other_op<F>(&mut self, other: &Self, mut f: F) where F: FnMut(B, B) -> B {
        // Unwrap BitArrays
        let self_bit_array = &mut self.bit_array;
        let other_bit_array = &other.bit_array;

        // virtually pad other with 0's for equal lengths
        let other_words = {
            let (_, result) = match_words(self_bit_array, other_bit_array);
            result
        };

        // Apply values found in other
        for (i, w) in other_words {
            let old = self_bit_array.storage()[i];
            let new = f(old, w);
            unsafe {
                self_bit_array.storage_mut()[i] = new;
            }
        }
    }

    /// Iterator over each usize stored in the `BitArraySet`.
    ///
    /// # Examples
    ///
    /// ```
    /// extern crate typenum;
    /// # extern crate bitarray_set;
    /// use typenum::{Unsigned, U8};
    /// use bitarray_set::BitArraySet;
    ///
    /// # fn main() {
    /// let s = BitArraySet::<u32, U8>::from_bytes(&[0b01001010]);
    ///
    /// // Print 1, 4, 6 in arbitrary order
    /// for x in s.iter() {
    ///     println!("{}", x);
    /// }
    /// # }
    /// ```
    #[inline]
    pub fn iter(&self) -> Iter<B> {
        Iter(BlockIter::from_blocks(self.bit_array.blocks()))
    }

    /// Iterator over each usize stored in `self` union `other`.
    /// See [union_with](#method.union_with) for an efficient in-place version.
    ///
    /// # Examples
    ///
    /// ```
    /// extern crate typenum;
    /// # extern crate bitarray_set;
    /// use typenum::{Unsigned, U8};
    /// use bitarray_set::BitArraySet;
    ///
    /// # fn main() {
    /// let a = BitArraySet::<u32, U8>::from_bytes(&[0b01101000]);
    /// let b = BitArraySet::<u32, U8>::from_bytes(&[0b10100000]);
    ///
    /// // Print 0, 1, 2, 4 in arbitrary order
    /// for x in a.union(&b) {
    ///     println!("{}", x);
    /// }
    /// # }
    /// ```
    #[inline]
    pub fn union<'a>(&'a self, other: &'a Self) -> Union<'a, B> {
        fn or<B: BitBlock>(w1: B, w2: B) -> B { w1 | w2 }

        Union(BlockIter::from_blocks(TwoBitPositions {
            set: self.bit_array.blocks(),
            other: other.bit_array.blocks(),
            merge: or,
        }))
    }

    /// Iterator over each usize stored in `self` intersect `other`.
    /// See [intersect_with](#method.intersect_with) for an efficient in-place version.
    ///
    /// # Examples
    ///
    /// ```
    /// extern crate typenum;
    /// # extern crate bitarray_set;
    /// use typenum::{Unsigned, U8};
    /// use bitarray_set::BitArraySet;
    ///
    /// # fn main() {
    /// let a = BitArraySet::<u32, U8>::from_bytes(&[0b01101000]);
    /// let b = BitArraySet::<u32, U8>::from_bytes(&[0b10100000]);
    ///
    /// // Print 2
    /// for x in a.intersection(&b) {
    ///     println!("{}", x);
    /// }
    /// # }
    /// ```
    #[inline]
    pub fn intersection<'a>(&'a self, other: &'a Self) -> Intersection<'a, B> {
        fn bitand<B: BitBlock>(w1: B, w2: B) -> B { w1 & w2 }
        let min = cmp::min(self.bit_array.len(), other.bit_array.len());

        Intersection(BlockIter::from_blocks(TwoBitPositions {
            set: self.bit_array.blocks(),
            other: other.bit_array.blocks(),
            merge: bitand,
        }).take(min))
    }

    /// Iterator over each usize stored in the `self` setminus `other`.
    /// See [difference_with](#method.difference_with) for an efficient in-place version.
    ///
    /// # Examples
    ///
    /// ```
    /// extern crate typenum;
    /// # extern crate bitarray_set;
    /// use typenum::{Unsigned, U8};
    /// use bitarray_set::BitArraySet;
    ///
    /// # fn main() {
    /// let a = BitArraySet::<u32, U8>::from_bytes(&[0b01101000]);
    /// let b = BitArraySet::<u32, U8>::from_bytes(&[0b10100000]);
    ///
    /// // Print 1, 4 in arbitrary order
    /// for x in a.difference(&b) {
    ///     println!("{}", x);
    /// }
    ///
    /// // Note that difference is not symmetric,
    /// // and `b - a` means something else.
    /// // This prints 0
    /// for x in b.difference(&a) {
    ///     println!("{}", x);
    /// }
    /// # }
    /// ```
    #[inline]
    pub fn difference<'a>(&'a self, other: &'a Self) -> Difference<'a, B> {
        fn diff<B: BitBlock>(w1: B, w2: B) -> <B as std::ops::BitAnd>::Output { w1 & !w2 }

        Difference(BlockIter::from_blocks(TwoBitPositions {
            set: self.bit_array.blocks(),
            other: other.bit_array.blocks(),
            merge: diff,
        }))
    }

    /// Iterator over each usize stored in the symmetric difference of `self` and `other`.
    /// See [symmetric_difference_with](#method.symmetric_difference_with) for
    /// an efficient in-place version.
    ///
    /// # Examples
    ///
    /// ```
    /// extern crate typenum;
    /// # extern crate bitarray_set;
    /// use typenum::{Unsigned, U8};
    /// use bitarray_set::BitArraySet;
    ///
    /// # fn main() {
    /// let a = BitArraySet::<u32, U8>::from_bytes(&[0b01101000]);
    /// let b = BitArraySet::<u32, U8>::from_bytes(&[0b10100000]);
    ///
    /// // Print 0, 1, 4 in arbitrary order
    /// for x in a.symmetric_difference(&b) {
    ///     println!("{}", x);
    /// }
    /// # }
    /// ```
    #[inline]
    pub fn symmetric_difference<'a>(&'a self, other: &'a Self) -> SymmetricDifference<'a, B> {
        fn bitxor<B: BitBlock>(w1: B, w2: B) -> B { w1 ^ w2 }

        SymmetricDifference(BlockIter::from_blocks(TwoBitPositions {
            set: self.bit_array.blocks(),
            other: other.bit_array.blocks(),
            merge: bitxor,
        }))
    }

    /// Unions in-place with the specified other bit array.
    ///
    /// # Examples
    ///
    /// ```
    /// extern crate typenum;
    /// # extern crate bitarray_set;
    /// use typenum::{Unsigned, U8};
    /// use bitarray_set::BitArraySet;
    ///
    /// # fn main() {
    /// let a   = 0b01101000;
    /// let b   = 0b10100000;
    /// let res = 0b11101000;
    ///
    /// let mut a = BitArraySet::<u32, U8>::from_bytes(&[a]);
    /// let b = BitArraySet::<u32, U8>::from_bytes(&[b]);
    /// let res = BitArraySet::<u32, U8>::from_bytes(&[res]);
    ///
    /// a.union_with(&b);
    /// assert_eq!(a, res);
    /// # }
    /// ```
    #[inline]
    pub fn union_with(&mut self, other: &Self) {
        self.other_op(other, |w1, w2| w1 | w2);
    }

    /// Intersects in-place with the specified other bit array.
    ///
    /// # Examples
    ///
    /// ```
    /// extern crate typenum;
    /// # extern crate bitarray_set;
    /// use typenum::{Unsigned, U8};
    /// use bitarray_set::BitArraySet;
    ///
    /// # fn main() {
    /// let a   = 0b01101000;
    /// let b   = 0b10100000;
    /// let res = 0b00100000;
    ///
    /// let mut a = BitArraySet::<u32, U8>::from_bytes(&[a]);
    /// let b = BitArraySet::<u32, U8>::from_bytes(&[b]);
    /// let res = BitArraySet::<u32, U8>::from_bytes(&[res]);
    ///
    /// a.intersect_with(&b);
    /// assert_eq!(a, res);
    /// # }
    /// ```
    #[inline]
    pub fn intersect_with(&mut self, other: &Self) {
        self.other_op(other, |w1, w2| w1 & w2);
    }

    /// Makes this bit array the difference with the specified other bit array
    /// in-place.
    ///
    /// # Examples
    ///
    /// ```
    /// extern crate typenum;
    /// # extern crate bitarray_set;
    /// use typenum::{Unsigned, U8};
    /// use bitarray_set::BitArraySet;
    ///
    /// # fn main() {
    /// let a   = 0b01101000;
    /// let b   = 0b10100000;
    /// let a_b = 0b01001000; // a - b
    /// let b_a = 0b10000000; // b - a
    ///
    /// let mut bva = BitArraySet::<u32, U8>::from_bytes(&[a]);
    /// let bvb = BitArraySet::<u32, U8>::from_bytes(&[b]);
    /// let bva_b = BitArraySet::<u32, U8>::from_bytes(&[a_b]);
    /// let bvb_a = BitArraySet::<u32, U8>::from_bytes(&[b_a]);
    ///
    /// bva.difference_with(&bvb);
    /// assert_eq!(bva, bva_b);
    ///
    /// let bva = BitArraySet::<u32, U8>::from_bytes(&[a]);
    /// let mut bvb = BitArraySet::<u32, U8>::from_bytes(&[b]);
    ///
    /// bvb.difference_with(&bva);
    /// assert_eq!(bvb, bvb_a);
    /// # }
    /// ```
    #[inline]
    pub fn difference_with(&mut self, other: &Self) {
        self.other_op(other, |w1, w2| w1 & !w2);
    }

    /// Makes this bit array the symmetric difference with the specified other
    /// bit array in-place.
    ///
    /// # Examples
    ///
    /// ```
    /// extern crate typenum;
    /// # extern crate bitarray_set;
    /// use typenum::{Unsigned, U8};
    /// use bitarray_set::BitArraySet;
    ///
    /// # fn main() {
    /// let a   = 0b01101000;
    /// let b   = 0b10100000;
    /// let res = 0b11001000;
    ///
    /// let mut a = BitArraySet::<u32, U8>::from_bytes(&[a]);
    /// let b = BitArraySet::<u32, U8>::from_bytes(&[b]);
    /// let res = BitArraySet::<u32, U8>::from_bytes(&[res]);
    ///
    /// a.symmetric_difference_with(&b);
    /// assert_eq!(a, res);
    /// # }
    /// ```
    #[inline]
    pub fn symmetric_difference_with(&mut self, other: &Self) {
        self.other_op(other, |w1, w2| w1 ^ w2);
    }

    /// Returns the number of set bits in this set.
    #[inline]
    pub fn len(&self) -> usize  {
        self.bit_array.blocks().fold(0, |acc, n| acc + n.count_ones() as usize)
    }

    /// Returns whether there are no bits set in this set
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.bit_array.none()
    }

    /// Clears all bits in this set
    #[inline]
    pub fn clear(&mut self) {
        self.bit_array.clear();
    }

    /// Returns `true` if this set contains the specified integer.
    #[inline]
    pub fn contains(&self, value: usize) -> bool {
        let bit_array = &self.bit_array;
        value < bit_array.len() && bit_array[value]
    }

    /// Returns `true` if the set has no elements in common with `other`.
    /// This is equivalent to checking for an empty intersection.
    #[inline]
    pub fn is_disjoint(&self, other: &Self) -> bool {
        self.intersection(other).next().is_none()
    }

    /// Returns `true` if the set is a subset of another.
    #[inline]
    pub fn is_subset(&self, other: &Self) -> bool {
        let self_bit_array = &self.bit_array;
        let other_bit_array = &other.bit_array;

        // Check that `self` intersect `other` is self
        self_bit_array.blocks().zip(other_bit_array.blocks()).all(|(w1, w2)| w1 & w2 == w1)
    }

    /// Returns `true` if the set is a superset of another.
    #[inline]
    pub fn is_superset(&self, other: &Self) -> bool {
        other.is_subset(self)
    }

    /// Adds a value to the set. Returns `true` if the value was not already
    /// present in the set.
    pub fn insert(&mut self, value: usize) -> bool {
        assert!(value <= NBits::to_usize(), "BitArraySet can only handle {:?} entries. Insert to {:?} requested.", NBits::to_usize(), value);
        if self.contains(value) {
            return false;
        }

        self.bit_array.set(value, true);
        return true;
    }

    /// Removes a value from the set. Returns `true` if the value was
    /// present in the set.
    pub fn remove(&mut self, value: usize) -> bool {
        if !self.contains(value) {
            return false;
        }

        self.bit_array.set(value, false);

        return true;
    }
}

impl<B: BitsIn + BitBlock + Default, NBits: Unsigned + NonZero> fmt::Debug for BitArraySet<B, NBits> 
    where NBits: Add<<B as BitsIn>::Output>,
    <NBits as Add<<B as BitsIn>::Output>>::Output: Sub<typenum::B1>,
    <<NBits as Add<<B as BitsIn>::Output>>::Output as Sub<typenum::B1>>::Output: Div<<B as BitsIn>::Output>,
    <<<NBits as Add<<B as BitsIn>::Output>>::Output as Sub<typenum::B1>>::Output as Div<<B as BitsIn>::Output>>::Output: generic_array::ArrayLength<B>
{
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        fmt.debug_set().entries(self).finish()
    }
}

impl<B: BitsIn + BitBlock + Default, NBits: Unsigned + NonZero> hash::Hash for BitArraySet<B, NBits> 
    where NBits: Add<<B as BitsIn>::Output>,
    <NBits as Add<<B as BitsIn>::Output>>::Output: Sub<typenum::B1>,
    <<NBits as Add<<B as BitsIn>::Output>>::Output as Sub<typenum::B1>>::Output: Div<<B as BitsIn>::Output>,
    <<<NBits as Add<<B as BitsIn>::Output>>::Output as Sub<typenum::B1>>::Output as Div<<B as BitsIn>::Output>>::Output: generic_array::ArrayLength<B>
{
    fn hash<H: hash::Hasher>(&self, state: &mut H) {
        for pos in self {
            pos.hash(state);
        }
    }
}

#[derive(Clone)]
struct BlockIter<T, B> {
    head: B,
    head_offset: usize,
    tail: T,
}

impl<T, B: BitBlock> BlockIter<T, B> where T: Iterator<Item=B> {
    fn from_blocks(mut blocks: T) -> BlockIter<T, B> {
        let h = blocks.next().unwrap_or(B::zero());
        BlockIter {tail: blocks, head: h, head_offset: 0}
    }
}

/// An iterator combining two `BitArraySet` iterators.
#[derive(Clone)]
struct TwoBitPositions<'a, B: 'a> {
    set: Blocks<'a, B>,
    other: Blocks<'a, B>,
    merge: fn(B, B) -> B,
}

/// An iterator for `BitArraySet`.
#[derive(Clone)]
pub struct Iter<'a, B: 'a>(BlockIter<Blocks<'a, B>, B>);
#[derive(Clone)]
pub struct Union<'a, B: 'a>(BlockIter<TwoBitPositions<'a, B>, B>);
#[derive(Clone)]
pub struct Intersection<'a, B: 'a>(Take<BlockIter<TwoBitPositions<'a, B>, B>>);
#[derive(Clone)]
pub struct Difference<'a, B: 'a>(BlockIter<TwoBitPositions<'a, B>, B>);
#[derive(Clone)]
pub struct SymmetricDifference<'a, B: 'a>(BlockIter<TwoBitPositions<'a, B>, B>);

impl<'a, T, B: BitBlock> Iterator for BlockIter<T, B> where T: Iterator<Item=B> {
    type Item = usize;

    fn next(&mut self) -> Option<usize> {
        while self.head == B::zero() {
            match self.tail.next() {
                Some(w) => self.head = w,
                None => return None
            }
            self.head_offset += B::bits();
        }

        // from the current block, isolate the
        // LSB and subtract 1, producing k:
        // a block with a number of set bits
        // equal to the index of the LSB
        let k = (self.head & (!self.head + B::one())) - B::one();
        // update block, removing the LSB
        self.head = self.head & (self.head - B::one());
        // return offset + (index of LSB)
        Some(self.head_offset + (B::count_ones(k) as usize))
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        match self.tail.size_hint() {
            (_, Some(h)) => (0, Some(1 + h * B::bits())),
            _ => (0, None)
        }
    }
}

impl<'a, B: BitBlock> Iterator for TwoBitPositions<'a, B> {
    type Item = B;

    fn next(&mut self) -> Option<B> {
        match (self.set.next(), self.other.next()) {
            (Some(a), Some(b)) => Some((self.merge)(a, b)),
            (Some(a), None) => Some((self.merge)(a, B::zero())),
            (None, Some(b)) => Some((self.merge)(B::zero(), b)),
            _ => return None
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let (a, au) = self.set.size_hint();
        let (b, bu) = self.other.size_hint();

        let upper = match (au, bu) {
            (Some(au), Some(bu)) => Some(cmp::max(au, bu)),
            _ => None
        };

        (cmp::max(a, b), upper)
    }
}

impl<'a, B: BitBlock> Iterator for Iter<'a, B> {
    type Item = usize;

    #[inline] fn next(&mut self) -> Option<usize> { self.0.next() }
    #[inline] fn size_hint(&self) -> (usize, Option<usize>) { self.0.size_hint() }
}

impl<'a, B: BitBlock> Iterator for Union<'a, B> {
    type Item = usize;

    #[inline] fn next(&mut self) -> Option<usize> { self.0.next() }
    #[inline] fn size_hint(&self) -> (usize, Option<usize>) { self.0.size_hint() }
}

impl<'a, B: BitBlock> Iterator for Intersection<'a, B> {
    type Item = usize;

    #[inline] fn next(&mut self) -> Option<usize> { self.0.next() }
    #[inline] fn size_hint(&self) -> (usize, Option<usize>) { self.0.size_hint() }
}

impl<'a, B: BitBlock> Iterator for Difference<'a, B> {
    type Item = usize;

    #[inline] fn next(&mut self) -> Option<usize> { self.0.next() }
    #[inline] fn size_hint(&self) -> (usize, Option<usize>) { self.0.size_hint() }
}

impl<'a, B: BitBlock> Iterator for SymmetricDifference<'a, B> {
    type Item = usize;

    #[inline] fn next(&mut self) -> Option<usize> { self.0.next() }
    #[inline] fn size_hint(&self) -> (usize, Option<usize>) { self.0.size_hint() }
}

impl<'a, B: BitsIn + BitBlock + Default, NBits: Unsigned + NonZero> IntoIterator for &'a BitArraySet<B, NBits> 
    where NBits: Add<<B as BitsIn>::Output>,
    <NBits as Add<<B as BitsIn>::Output>>::Output: Sub<typenum::B1>,
    <<NBits as Add<<B as BitsIn>::Output>>::Output as Sub<typenum::B1>>::Output: Div<<B as BitsIn>::Output>,
    <<<NBits as Add<<B as BitsIn>::Output>>::Output as Sub<typenum::B1>>::Output as Div<<B as BitsIn>::Output>>::Output: generic_array::ArrayLength<B>
{
    type Item = usize;
    type IntoIter = Iter<'a, B>;

    fn into_iter(self) -> Iter<'a, B> {
        self.iter()
    }
}

#[cfg(test)]
mod tests {
    use std::cmp::Ordering::{Equal, Greater, Less};
    use super::BitArraySet;
    use bit_array::BitArray;
    use typenum::{U4, U8, U10, U51, U64, U100, U104, U152, U201, U221, U401, U501, U1001};

    #[test]
    fn test_bit_set_show() {
        let mut s = BitArraySet::<u32, U51>::new();
        s.insert(1);
        s.insert(10);
        s.insert(50);
        s.insert(2);
        assert_eq!("{1, 2, 10, 50}", format!("{:?}", s));
    }

    #[test]
    fn test_bit_set_from_usizes() {
        let usizes = vec![0, 2, 2, 3];
        let a: BitArraySet<u32, U4> = usizes.into_iter().collect();
        let mut b = BitArraySet::new();
        b.insert(0);
        b.insert(2);
        b.insert(3);
        assert_eq!(a, b);
    }

    #[test]
    fn test_bit_set_iterator() {
        let usizes = vec![0, 2, 2, 3];
        let bit_array: BitArraySet<u32, U4> = usizes.into_iter().collect();

        let idxs: Vec<_> = bit_array.iter().collect();
        assert_eq!(idxs, [0, 2, 3]);

        let long: BitArraySet<u32, U1001> = (0..1000).filter(|&n| n % 2 == 0).collect();
        let real: Vec<_> = (0..1000/2).map(|x| x*2).collect();

        let idxs: Vec<_> = long.iter().collect();
        assert_eq!(idxs, real);
    }

    #[test]
    fn test_bit_set_frombit_array_init() {
        let bools = [true, false];
        for &b in &bools {
            let bitset_10 = BitArraySet::from_bit_array(BitArray::<u32, U10>::from_elem(b));
            let bitset_64 = BitArraySet::from_bit_array(BitArray::<u32, U64>::from_elem(b));
            let bitset_100 = BitArraySet::from_bit_array(BitArray::<u32, U100>::from_elem(b));

            assert_eq!(bitset_10.contains(1), b);
            assert_eq!(bitset_10.contains((9)), b);
            assert!(!bitset_10.contains(10));

            assert_eq!(bitset_64.contains(1), b);
            assert_eq!(bitset_64.contains((63)), b);
            assert!(!bitset_64.contains(64));

            assert_eq!(bitset_100.contains(1), b);
            assert_eq!(bitset_100.contains((99)), b);
            assert!(!bitset_100.contains(100));
        }
    }

    #[test]
    fn test_bit_array_masking() {
        let b: BitArray<u32, U152> = BitArray::from_elem(true);
        let mut bs = BitArraySet::from_bit_array(b);
        bs.remove(140);
        bs.remove(149);
        bs.remove(150);
        bs.remove(151);
        assert!(bs.contains(139));
        assert!(!bs.contains(140));
        assert!(bs.insert(150));
        assert!(!bs.contains(140));
        assert!(!bs.contains(149));
        assert!(bs.contains(150));
        assert!(!bs.contains(151));
    }

    #[test]
    fn test_bit_set_basic() {
        let mut b = BitArraySet::<u32, U401>::new();
        assert!(b.insert(3));
        assert!(!b.insert(3));
        assert!(b.contains(3));
        assert!(b.insert(4));
        assert!(!b.insert(4));
        assert!(b.contains(3));
        assert!(b.insert(400));
        assert!(!b.insert(400));
        assert!(b.contains(400));
        assert_eq!(b.len(), 3);
    }

    #[test]
    fn test_bit_set_intersection() {
        let mut a = BitArraySet::<u32, U104>::new();
        let mut b = BitArraySet::<u32, U104>::new();

        assert!(a.insert(11));
        assert!(a.insert(1));
        assert!(a.insert(3));
        assert!(a.insert(77));
        assert!(a.insert(103));
        assert!(a.insert(5));

        assert!(b.insert(2));
        assert!(b.insert(11));
        assert!(b.insert(77));
        assert!(b.insert(5));
        assert!(b.insert(3));

        let expected = [3, 5, 11, 77];
        let actual: Vec<_> = a.intersection(&b).collect();
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_bit_set_difference() {
        let mut a = BitArraySet::<u32, U501>::new();
        let mut b = BitArraySet::<u32, U501>::new();

        assert!(a.insert(1));
        assert!(a.insert(3));
        assert!(a.insert(5));
        assert!(a.insert(200));
        assert!(a.insert(500));

        assert!(b.insert(3));
        assert!(b.insert(200));

        let expected = [1, 5, 500];
        let actual: Vec<_> = a.difference(&b).collect();
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_bit_set_symmetric_difference() {
        let mut a = BitArraySet::<u32, U221>::new();
        let mut b = BitArraySet::<u32, U221>::new();

        assert!(a.insert(1));
        assert!(a.insert(3));
        assert!(a.insert(5));
        assert!(a.insert(9));
        assert!(a.insert(11));

        assert!(b.insert(3));
        assert!(b.insert(9));
        assert!(b.insert(14));
        assert!(b.insert(220));

        let expected = [1, 5, 11, 14, 220];
        let actual: Vec<_> = a.symmetric_difference(&b).collect();
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_bit_set_union() {
        let mut a = BitArraySet::<u32, U201>::new();
        let mut b = BitArraySet::<u32, U201>::new();
        assert!(a.insert(1));
        assert!(a.insert(3));
        assert!(a.insert(5));
        assert!(a.insert(9));
        assert!(a.insert(11));
        assert!(a.insert(160));
        assert!(a.insert(19));
        assert!(a.insert(24));
        assert!(a.insert(200));

        assert!(b.insert(1));
        assert!(b.insert(5));
        assert!(b.insert(9));
        assert!(b.insert(13));
        assert!(b.insert(19));

        let expected = [1, 3, 5, 9, 11, 13, 19, 24, 160, 200];
        let actual: Vec<_> = a.union(&b).collect();
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_bit_set_subset() {
        let mut set1 = BitArraySet::<u32, U401>::new();
        let mut set2 = BitArraySet::<u32, U401>::new();

        assert!(set1.is_subset(&set2)); //  {}  {}
        set2.insert(100);
        assert!(set1.is_subset(&set2)); //  {}  { 1 }
        set2.insert(200);
        assert!(set1.is_subset(&set2)); //  {}  { 1, 2 }
        set1.insert(200);
        assert!(set1.is_subset(&set2)); //  { 2 }  { 1, 2 }
        set1.insert(300);
        assert!(!set1.is_subset(&set2)); // { 2, 3 }  { 1, 2 }
        set2.insert(300);
        assert!(set1.is_subset(&set2)); // { 2, 3 }  { 1, 2, 3 }
        set2.insert(400);
        assert!(set1.is_subset(&set2)); // { 2, 3 }  { 1, 2, 3, 4 }
        set2.remove(100);
        assert!(set1.is_subset(&set2)); // { 2, 3 }  { 2, 3, 4 }
        set2.remove(300);
        assert!(!set1.is_subset(&set2)); // { 2, 3 }  { 2, 4 }
        set1.remove(300);
        assert!(set1.is_subset(&set2)); // { 2 }  { 2, 4 }
    }

    #[test]
    fn test_bit_set_is_disjoint() {
        let a = BitArraySet::<u32, U8>::from_bytes(&[0b10100010]);
        let b = BitArraySet::<u32, U8>::from_bytes(&[0b01000000]);
        let c = BitArraySet::<u32, U8>::new();
        let d = BitArraySet::<u32, U8>::from_bytes(&[0b00110000]);

        assert!(!a.is_disjoint(&d));
        assert!(!d.is_disjoint(&a));

        assert!(a.is_disjoint(&b));
        assert!(a.is_disjoint(&c));
        assert!(b.is_disjoint(&a));
        assert!(b.is_disjoint(&c));
        assert!(c.is_disjoint(&a));
        assert!(c.is_disjoint(&b));
    }

    #[test]
    fn test_bit_set_union_with() {
        //a should grow to include larger elements
        let mut a = BitArraySet::<u32, U8>::new();
        a.insert(0);
        let mut b = BitArraySet::<u32, U8>::new();
        b.insert(5);
        let expected = BitArraySet::<u32, U8>::from_bytes(&[0b10000100]);
        a.union_with(&b);
        assert_eq!(a, expected);

        // Standard
        let mut a = BitArraySet::<u32, U8>::from_bytes(&[0b10100010]);
        let mut b = BitArraySet::<u32, U8>::from_bytes(&[0b01100010]);
        let c = a.clone();
        a.union_with(&b);
        b.union_with(&c);
        assert_eq!(a.len(), 4);
        assert_eq!(b.len(), 4);
    }

    #[test]
    fn test_bit_set_intersect_with() {
        // Explicitly 0'ed bits
        let mut a = BitArraySet::<u32, U8>::from_bytes(&[0b10100010]);
        let mut b = BitArraySet::<u32, U8>::from_bytes(&[0b00000000]);
        let c = a.clone();
        a.intersect_with(&b);
        b.intersect_with(&c);
        assert!(a.is_empty());
        assert!(b.is_empty());

        // Uninitialized bits should behave like 0's
        let mut a = BitArraySet::<u32, U8>::from_bytes(&[0b10100010]);
        let mut b = BitArraySet::<u32, U8>::new();
        let c = a.clone();
        a.intersect_with(&b);
        b.intersect_with(&c);
        assert!(a.is_empty());
        assert!(b.is_empty());

        // Standard
        let mut a = BitArraySet::<u32, U8>::from_bytes(&[0b10100010]);
        let mut b = BitArraySet::<u32, U8>::from_bytes(&[0b01100010]);
        let c = a.clone();
        a.intersect_with(&b);
        b.intersect_with(&c);
        assert_eq!(a.len(), 2);
        assert_eq!(b.len(), 2);
    }

    #[test]
    fn test_bit_set_difference_with() {
        // Explicitly 0'ed bits
        let mut a = BitArraySet::<u32, U8>::from_bytes(&[0b00000000]);
        let b = BitArraySet::<u32, U8>::from_bytes(&[0b10100010]);
        a.difference_with(&b);
        assert!(a.is_empty());

        // Uninitialized bits should behave like 0's
        let mut a = BitArraySet::<u32, U8>::new();
        let b = BitArraySet::<u32, U8>::from_bytes(&[0b11111111]);
        a.difference_with(&b);
        assert!(a.is_empty());

        // Standard
        let mut a = BitArraySet::<u32, U8>::from_bytes(&[0b10100010]);
        let mut b = BitArraySet::<u32, U8>::from_bytes(&[0b01100010]);
        let c = a.clone();
        a.difference_with(&b);
        b.difference_with(&c);
        assert_eq!(a.len(), 1);
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_bit_set_symmetric_difference_with() {
        //a should grow to include larger elements
        let mut a = BitArraySet::<u32, U8>::new();
        a.insert(0);
        a.insert(1);
        let mut b = BitArraySet::<u32, U8>::new();
        b.insert(1);
        b.insert(5);
        let expected = BitArraySet::<u32, U8>::from_bytes(&[0b10000100]);
        a.symmetric_difference_with(&b);
        assert_eq!(a, expected);

        let mut a = BitArraySet::<u32, U8>::from_bytes(&[0b10100010]);
        let b = BitArraySet::<u32, U8>::new();
        let c = a.clone();
        a.symmetric_difference_with(&b);
        assert_eq!(a, c);

        // Standard
        let mut a = BitArraySet::<u32, U8>::from_bytes(&[0b11100010]);
        let mut b = BitArraySet::<u32, U8>::from_bytes(&[0b01101010]);
        let c = a.clone();
        a.symmetric_difference_with(&b);
        b.symmetric_difference_with(&c);
        assert_eq!(a.len(), 2);
        assert_eq!(b.len(), 2);
    }

    #[test]
    fn test_bit_set_eq() {
        let a = BitArraySet::<u32, U8>::from_bytes(&[0b10100010]);
        let b = BitArraySet::<u32, U8>::from_bytes(&[0b00000000]);
        let c = BitArraySet::<u32, U8>::new();

        assert!(a == a);
        assert!(a != b);
        assert!(a != c);
        assert!(b == b);
        assert!(b == c);
        assert!(c == c);
    }

    #[test]
    fn test_bit_set_cmp() {
        let a = BitArraySet::<u32, U8>::from_bytes(&[0b10100010]);
        let b = BitArraySet::<u32, U8>::from_bytes(&[0b00000000]);
        let c = BitArraySet::<u32, U8>::new();

        assert_eq!(a.cmp(&b), Greater);
        assert_eq!(a.cmp(&c), Greater);
        assert_eq!(b.cmp(&a), Less);
        assert_eq!(b.cmp(&c), Equal);
        assert_eq!(c.cmp(&a), Less);
        assert_eq!(c.cmp(&b), Equal);
    }

    #[test]
    fn test_bit_array_remove() {
        let mut a = BitArraySet::<u32, U1001>::new();

        assert!(a.insert(1));
        assert!(a.remove(1));

        assert!(a.insert(100));
        assert!(a.remove(100));

        assert!(a.insert(1000));
        assert!(a.remove(1000));
    }

    #[test]
    fn test_bit_array_clone() {
        let mut a = BitArraySet::<u32, U1001>::new();

        assert!(a.insert(1));
        assert!(a.insert(100));
        assert!(a.insert(1000));

        let mut b = a.clone();

        assert!(a == b);

        assert!(b.remove(1));
        assert!(a.contains(1));

        assert!(a.remove(1000));
        assert!(b.contains(1000));
    }

/*
    #[test]
    fn test_bit_set_append() {
        let mut a = BitArraySet::new();
        a.insert(2);
        a.insert(6);

        let mut b = BitArraySet::new();
        b.insert(1);
        b.insert(3);
        b.insert(6);

        a.append(&mut b);

        assert_eq!(a.len(), 4);
        assert_eq!(b.len(), 0);
        assert!(b.capacity() >= 6);

        assert_eq!(a, BitArraySet::from_bytes(&[0b01110010]));
    }

    #[test]
    fn test_bit_set_split_off() {
        // Split at 0
        let mut a = BitArraySet::from_bytes(&[0b10100000, 0b00010010, 0b10010010,
                                         0b00110011, 0b01101011, 0b10101101]);

        let b = a.split_off(0);

        assert_eq!(a.len(), 0);
        assert_eq!(b.len(), 21);

        assert_eq!(b, BitArraySet::from_bytes(&[0b10100000, 0b00010010, 0b10010010,
                                           0b00110011, 0b01101011, 0b10101101]);

        // Split behind last element
        let mut a = BitArraySet::from_bytes(&[0b10100000, 0b00010010, 0b10010010,
                                         0b00110011, 0b01101011, 0b10101101]);

        let b = a.split_off(50);

        assert_eq!(a.len(), 21);
        assert_eq!(b.len(), 0);

        assert_eq!(a, BitArraySet::from_bytes(&[0b10100000, 0b00010010, 0b10010010,
                                           0b00110011, 0b01101011, 0b10101101]));

        // Split at arbitrary element
        let mut a = BitArraySet::from_bytes(&[0b10100000, 0b00010010, 0b10010010,
                                         0b00110011, 0b01101011, 0b10101101]);

        let b = a.split_off(34);

        assert_eq!(a.len(), 12);
        assert_eq!(b.len(), 9);

        assert_eq!(a, BitArraySet::from_bytes(&[0b10100000, 0b00010010, 0b10010010,
                                           0b00110011, 0b01000000]));
        assert_eq!(b, BitArraySet::from_bytes(&[0, 0, 0, 0,
                                           0b00101011, 0b10101101]));
    }
*/
}

#[cfg(all(test, feature = "nightly"))]
mod bench {
    use super::BitArraySet;
    use bit_array::BitArray;
    use rand::{Rng, thread_rng, ThreadRng};
    use typenum::*;

    use test::{Bencher, black_box};

    //const BENCH_BITS: usize = 1 << 14;
    const BITS: usize = 32;

    fn rng() -> ThreadRng {
        thread_rng()
    }

    #[bench]
    fn bench_bit_arrayset_small(b: &mut Bencher) {
        let mut r = rng();
        let mut bit_arrayset = BitArraySet::<u32, U32>::new();
        b.iter(|| {
            for _ in 0..100 {
                bit_arrayset.insert((r.next_u32() as usize) % BITS);
            }
            black_box(&bit_arrayset);
        });
    }

    #[bench]
    fn bench_bit_arrayset_big(b: &mut Bencher) {
        let mut r = rng();
        let mut bit_arrayset = BitArraySet::<u32, U16384>::new();
        b.iter(|| {
            for _ in 0..100 {
                bit_arrayset.insert((r.next_u32() as usize) % U16384::to_usize());
            }
            black_box(&bit_arrayset);
        });
    }

    #[bench]
    fn bench_bit_arrayset_iter(b: &mut Bencher) {
        let bit_arrayset = BitArraySet::from_bit_array(BitArray::<u32, U32>::from_fn(
                                              |idx| {idx % 3 == 0}));
        b.iter(|| {
            let mut sum = 0;
            for idx in &bit_arrayset {
                sum += idx as usize;
            }
            sum
        })
    }
}
