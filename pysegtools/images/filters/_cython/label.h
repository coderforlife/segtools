// Cython helper header for label filters.
// Implements various high-efficiency algorithms

#pragma once
#include "npy_helper.h"

// The publicly available functions are as follows
// Replace: using the mapping keys->vals to convert src->dst
template <typename K, typename V> inline void map_replace(const K* keys, const V* vals, const intptr_t nkeys, const K* src, V* dst, const intptr_t nrows);
template <typename K, typename V> inline void map_replace_rows(const K* keys, const V* vals, const intptr_t nkeys, const K* src, V* dst, const intptr_t nrows, const intptr_t ncols);
// Renumber: convert src->dst by assigning every unique element in src an integer
template <typename K, typename V> inline V map_renumber(const K* src, V* dst, const intptr_t size);
template <typename K, typename V> inline V map_renumber_rows(const K* src, V* dst, const intptr_t size, const intptr_t ncols);
// Set-Union: perform the union operation of two sets (sorted, unique, arrays) (see std::set_union for more details)
template <typename T> inline intptr_t set_union(const T* first1, const T* last1, const T* first2, const T* last2, T* out);
template <typename T> inline intptr_t row_set_union(const T* first1, const T* last1, const T* first2, const T* last2, T* out, const intptr_t ncols);
// Merge-Sort-Unique: run merge sort on an array, removing duplicates as they are found
template <typename T> intptr_t merge_sort_unique(T* first, T* last);
template <typename T> inline intptr_t merge_sort_unique_rows(T* first, T* last, const intptr_t ncols);
// Row-Lower/Upper-Bound: run binary search on a sorted array (see std::lower_bound/std::upper_bound for more details); runs for multiple values
template <typename T> inline void row_lower_bounds(const T* sorted_first, const T* sorted_last, const T* first, const T* last, uintptr_t* out, const intptr_t ncols);
template <typename T> inline void row_upper_bounds(const T* sorted_first, const T* sorted_last, const T* first, const T* last, uintptr_t* out, const intptr_t ncols);


#include <algorithm> // std::copy, std::unique, std::unique_copy
#include <new>       // std::bad_alloc
#include <string.h>  // memcmp
#include <memory.h>  // malloc, free


//////////////////// Replace and Remove Functions ////////////////////
template <typename T>
class __row_zeros // holds a row of "zeros" that auto-delocates
{
    T* z;
public:
    inline __row_zeros(intptr_t ncols) : z((T*)memset(malloc(ncols*sizeof(T)), 0, ncols*sizeof(T)))
    {
        if (!this->z) { throw std::bad_alloc(); }
    }
    inline ~__row_zeros() { free(this->z); this->z = NULL; }
    inline const T* const zeros() const { return this->z; }
};

template <typename K, typename V>
inline void map_replace(const K* keys, const V* vals, const intptr_t nkeys, const K* src, V* dst, const intptr_t nrows)
{
    typedef std::unordered_map<K, V, npy_hash<K>, npy_hash<K> > map_t;
    map_t map(nkeys*2);
    for (intptr_t i = 0; i < nkeys; ++i) { map[keys[i]] = vals[i]; }
    for (intptr_t i = 0; i < nrows; ++i) { dst[i] = map[src[i]]; }
}
template <typename K, typename V>
inline void __map_replace_rows(const K* keys, const V* vals, const intptr_t nkeys, const K* src, V* dst, const intptr_t nrows, const intptr_t ncols)
{
    typedef std::unordered_map<const K*, V, row_hash<K>, row_hash<K> > map_t;
    map_t map(nkeys*2, row_hash<K>(ncols), row_hash<K>(ncols));
    for (intptr_t i = 0, j = 0; i < nkeys; ++i, j += ncols) { map[keys+j] = vals[i]; }
    for (intptr_t i = 0, j = 0; i < nrows; ++i, j += ncols) { dst[i] = map[src+j]; }
}
template <typename K, typename V, intptr_t ncols>
inline void __map_replace_rows(const K* keys, const V* vals, const intptr_t nkeys, const K* src, V* dst, const intptr_t nrows)
{
    typedef std::unordered_map<const K*, V, row_hash<K,ncols>, row_hash<K,ncols> > map_t;
    map_t map(nkeys*2);
    for (intptr_t i = 0, j = 0; i < nkeys; ++i, j += ncols) { map[keys+j] = vals[i]; }
    for (intptr_t i = 0, j = 0; i < nrows; ++i, j += ncols) { dst[i] = map[src+j]; }
}

template <typename K, typename V>
inline V map_renumber(const K* src, V* dst, const intptr_t size)
{
    typedef std::unordered_map<K, V, npy_hash<K>, npy_hash<K> > map_t;
    map_t map(1024);
    map[K()] = V();
    V n = V();
    for (intptr_t i = 0; i < size; ++i)
    {
        const map_t::const_iterator itr = map.find(src[i]);
        if (itr == map.end()) { dst[i] = ++n; map[src[i]] = n; }
        else { dst[i] = itr->second; }
    }
    return n;
}
template <typename K, typename V>
inline V __map_renumber_rows(const K* src, V* dst, const intptr_t size, const intptr_t ncols)
{
    typedef std::unordered_map<const K*, V, row_hash<K>, row_hash<K> > map_t;
    map_t map(1024, row_hash<K>(ncols), row_hash<K>(ncols));
    __row_zeros<K> rz(ncols);
    map[rz.zeros()] = V();
    V n = V();
    for (intptr_t i = 0, j = 0; i < size; ++i, j += ncols)
    {
        const map_t::const_iterator itr = map.find(src+j);
        if (itr == map.end()) { dst[i] = ++n; map[src+j] = n; }
        else { dst[i] = itr->second; }
    }
    return n;
}
template <typename K, typename V, intptr_t ncols>
inline V __map_renumber_rows(const K* src, V* dst, const intptr_t size)
{
    typedef std::unordered_map<const K*, V, row_hash<K,ncols>, row_hash<K,ncols> > map_t;
    map_t map(1024);
    K rz[ncols];
    memset(rz, 0, ncols*sizeof(K));
    map[rz] = V();
    V n = V();
    for (intptr_t i = 0, j = 0; i < size; ++i, j += ncols)
    {
        const map_t::const_iterator itr = map.find(src+j);
        if (itr == map.end()) { dst[i] = ++n; map[src+j] = n; }
        else { dst[i] = itr->second; }
    }
    return n;
}

// Choose a specialization based on the number of columns (currently 1-4 columns are specialized)
template <typename K, typename V>
inline void map_replace_rows(const K* keys, const V* vals, const intptr_t nkeys, const K* src, V* dst, const intptr_t nrows, const intptr_t ncols)
{
    switch (ncols)
    {
    case 1: return map_replace(keys, vals, nkeys, src, dst, nrows);
    case 2: return __map_replace_rows<K,V,2>(keys, vals, nkeys, src, dst, nrows);
    case 3: return __map_replace_rows<K,V,3>(keys, vals, nkeys, src, dst, nrows);
    case 4: return __map_replace_rows<K,V,4>(keys, vals, nkeys, src, dst, nrows);
    default: return __map_replace_rows(keys, vals, nkeys, src, dst, nrows, ncols);
    }
}
template <typename K, typename V>
inline V map_renumber_rows(const K* src, V* dst, const intptr_t size, const intptr_t ncols)
{
    switch (ncols)
    {
    case 1: return map_renumber(src, dst, size);
    case 2: return __map_renumber_rows<K,V,2>(src, dst, size);
    case 3: return __map_renumber_rows<K,V,3>(src, dst, size);
    case 4: return __map_renumber_rows<K,V,4>(src, dst, size);
    default: return __map_renumber_rows(src, dst, size, ncols);
    }
}


//////////////////// General Operations for working with Rows ////////////////////
// Compile-time number of columns
template <typename T, intptr_t ncols>
inline T* __row_unique(T* first, const T* last)
{
    typedef row_ops<T,ncols> ro;
    T* out = first;
    while ((first+=ncols) != last) { if (ro::ne(out, first)) { ro::cp((out+=ncols), first); } }
    return out + ncols;
}
template <typename T, intptr_t ncols>
inline T* __row_unique_copy(const T* first, const T* last, T* out)
{
    typedef row_ops<T,ncols> ro;
    ro::cp(out, first);
    while ((first+=ncols) != last) { if (ro::ne(out, first)) { ro::cp((out+=ncols), first); } }
    return out + ncols;
}
template <typename T, intptr_t ncols>
inline T* __row_set_union(const T* first1, const T* last1, const T* first2, const T* last2, T* out)
{
    typedef row_ops<T,ncols> ro;
    for (;;)
    {
        if (first1 == last1) { return std::copy(first2, last2, out); }
        if (first2 == last2) { return std::copy(first1, last1, out); }
        if      (ro::lt(first1, first2)) { ro::cp(out, first1); first1 += ncols; }
        else if (ro::lt(first2, first1)) { ro::cp(out, first2); first2 += ncols; }
        else                             { ro::cp(out, first1); first1 += ncols; first2 += ncols; }
        out += ncols;
    }
}

// Runtime number of columns
template <typename T>
inline T* __row_unique(T* first, const T* last, const intptr_t ncols)
{
    typedef row_ops<T> ro;
    T* out = first;
    while ((first+=ncols) != last) { if (ro::ne(out, first, ncols)) { ro::cp((out+=ncols), first, ncols); } }
    return out + ncols;
}
template <typename T>
inline T* __row_unique_copy(const T* first, const T* last, T* out, const intptr_t ncols)
{
    typedef row_ops<T> ro;
    ro::cp(out, first, ncols);
    while ((first+=ncols) != last) { if (ro::ne(out, first, ncols)) { ro::cp((out+=ncols), first, ncols); } }
    return out + ncols;
}
template <typename T>
inline T* __row_set_union(const T* first1, const T* last1, const T* first2, const T* last2, T* out, const intptr_t ncols)
{
    typedef row_ops<T> ro;
    for (;;)
    {
        if (first1 == last1) { return std::copy(first2, last2, out); }
        if (first2 == last2) { return std::copy(first1, last1, out); }
        if      (ro::lt(first1, first2, ncols)) { ro::cp(out, first1, ncols); first1 += ncols; }
        else if (ro::lt(first2, first1, ncols)) { ro::cp(out, first2, ncols); first2 += ncols; }
        else                                    { ro::cp(out, first1, ncols); first1 += ncols; first2 += ncols; }
        out += ncols;
    }
}
template <typename T>
inline T* __set_union(const T* first1, const T* last1, const T* first2, const T* last2, T* out)
{
    for (;;)
    {
        if (first1 == last1) { return std::copy(first2, last2, out); }
        if (first2 == last2) { return std::copy(first1, last1, out); }
        if      (sort_ops<T>::lt(*first1,*first2)) { *out = *first1++; }
        else if (sort_ops<T>::lt(*first2,*first1)) { *out = *first2++; }
        else                                       { *out = *first1++; ++first2; }
        ++out;
    }
}
// These functions make the above usable in Cython and chooses a specialization based on the number of columns (currently 1-4 columns are specialized)
template <typename T> inline intptr_t set_union(const T* first1, const T* last1, const T* first2, const T* last2, T* out) { return __set_union(first1, last1, first2, last2, out) - out; }
template <typename T> inline intptr_t row_set_union(const T* first1, const T* last1, const T* first2, const T* last2, T* out, const intptr_t ncols)
{
    switch (ncols)
    {
    case 1: return __set_union(first1, last1, first2, last2, out) - out;
    case 2: return (__row_set_union<T,2>(first1, last1, first2, last2, out) - out) / 2;
    case 3: return (__row_set_union<T,3>(first1, last1, first2, last2, out) - out) / 3;
    case 4: return (__row_set_union<T,4>(first1, last1, first2, last2, out) - out) / 4;
    default: return (__row_set_union<T>(first1, last1, first2, last2, out, ncols) - out) / ncols;
    }
}


//////////////////// Uniquifying Merge Sort ////////////////////
#define MERGE_SORT_THRESHOLD 64
template <typename T>
inline void __insertion_sort(T* first, T* last)
{
    for (T* a = first+1; a != last; ++a)
    {
        T x = *a, *b = a;
        while (b != first && sort_ops<T>::lt(x, *(b-1))) { *b = *(b-1); --b; }
        *b = x;
    }
}
template <typename T> T* __merge_sort_unique1(T* first, T* last, T* temp);
template <typename T>
T* __merge_sort_unique2(T* first, T* last, T* temp) // saves the output to first
{
    intptr_t N = last - first;
    if (N <= MERGE_SORT_THRESHOLD) { __insertion_sort(first, last); return std::unique(first, last); }
    T* mid = first + N / 2;
    T* a_last = __merge_sort_unique1(first, mid, temp);
    T* b_last = __merge_sort_unique1(mid, last, a_last);
    return __set_union(temp, a_last, a_last, b_last, first);
}
template <typename T>
T* __merge_sort_unique1(T* first, T* last, T* temp) // saves the output to temp
{
    intptr_t N = last - first;
    if (N <= MERGE_SORT_THRESHOLD) { __insertion_sort(first, last); return std::unique_copy(first, last, temp); }
    T* mid = first + N / 2;
    T* a_last = __merge_sort_unique2(first, mid, temp);
    T* b_last = __merge_sort_unique2(mid, last, temp);
    return __set_union(first, a_last, mid, b_last, temp);
}
template <typename T>
intptr_t merge_sort_unique(T* first, T* last)
{
    intptr_t N = last - first;
    if (N <= MERGE_SORT_THRESHOLD) { if (N == 0) { return 0; } __insertion_sort(first, last); return std::unique(first, last) - first; }
    T* temp = (T*)malloc((N+1)/2*sizeof(T));
    if (!temp) { throw std::bad_alloc(); }
    T* mid = first + N/2;
    T* b_last = __merge_sort_unique1(mid, last, temp); // second half goes into temporary
    T* a_last = __merge_sort_unique1(first, mid, mid); // first half goes into second half
    last = __set_union(mid, a_last, temp, b_last, first); // merge second half and temporary into first half (and eventually second half)
    free(temp);
    return last - first;
}



//////////////////// Uniquifying Merge Sort for Rows ////////////////////
// Compile-time number of columns
template <typename T, intptr_t ncols>
inline void __row_insertion_sort(T* first, const T* last)
{
    typedef row_ops<T,ncols> ro;
    T x[ncols];
    for (T* a = first+ncols; a != last; a += ncols)
    {
        ro::cp(x, a);
        T* b = a;
        while (b != first && ro::lt(x, b-ncols)) { ro::cp(b, b-ncols); b -= ncols; }
        ro::cp(b, x);
    }
}
template <typename T, intptr_t ncols> T* __merge_sort_unique_rows1(T* first, T* last, T* temp);
template <typename T, intptr_t ncols>
T* __merge_sort_unique_rows2(T* first, T* last, T* temp) // saves the output to first
{
    const intptr_t N = (last - first) / ncols;
    if (N <= MERGE_SORT_THRESHOLD) { __row_insertion_sort<T,ncols>(first, last); return __row_unique<T,ncols>(first, last); }
    T* mid = first + (N / 2) * ncols;
    T* a_last = __merge_sort_unique_rows1<T,ncols>(first, mid, temp);
    T* b_last = __merge_sort_unique_rows1<T,ncols>(mid, last, a_last);
    return __row_set_union<T,ncols>(temp, a_last, a_last, b_last, first);
}
template <typename T, intptr_t ncols>
T* __merge_sort_unique_rows1(T* first, T* last, T* temp) // saves the output to temp
{
    const intptr_t N = (last - first) / ncols;
    if (N <= MERGE_SORT_THRESHOLD) { __row_insertion_sort<T,ncols>(first, last); return __row_unique_copy<T,ncols>(first, last, temp); }
    T* mid = first + (N / 2) * ncols;
    T* a_last = __merge_sort_unique_rows2<T,ncols>(first, mid, temp);
    T* b_last = __merge_sort_unique_rows2<T,ncols>(mid, last, temp);
    return __row_set_union<T,ncols>(first, a_last, mid, b_last, temp);
}
template <typename T, intptr_t ncols>
intptr_t __merge_sort_unique_rows(T* first, T* last)
{
    const intptr_t N = (last - first) / ncols;
    if (N <= MERGE_SORT_THRESHOLD) { if (N == 0) { return 0; } __row_insertion_sort<T,ncols>(first, last); return (__row_unique<T,ncols>(first, last)-first)/ncols; }
    T* temp = (T*)malloc((N+1)/2*sizeof(T)*ncols);
    if (!temp) { throw std::bad_alloc(); }
    T* mid = first + (N/2)*ncols;
    T* b_last = __merge_sort_unique_rows1<T,ncols>(mid, last, temp); // second half goes into temporary
    T* a_last = __merge_sort_unique_rows1<T,ncols>(first, mid, mid); // first half goes into second half
    last = __row_set_union<T,ncols>(mid, a_last, temp, b_last, first); // merge second half and temporary into first half (and eventually second half)
    free(temp);
    return (last-first)/ncols;
}

// Runtime number of columns
template <typename T>
inline void __row_insertion_sort(T* first, const T* last, T* row, const intptr_t ncols)
{
    typedef row_ops<T> ro;
    for (T* a = first+ncols; a != last; a += ncols)
    {
        ro::cp(row, a, ncols);
        T* b = a;
        while (b != first && ro::lt(row, b-ncols, ncols)) { ro::cp(b, b-ncols, ncols); b -= ncols; }
        ro::cp(b, row, ncols);
    }
}
template <typename T> T* __merge_sort_unique_rows1(T* first, T* last, T* temp, T* row, const intptr_t ncols);
template <typename T>
T* __merge_sort_unique_rows2(T* first, T* last, T* temp, T* row, const intptr_t ncols) // saves the output to first
{
    const intptr_t N = (last - first) / ncols;
    if (N <= MERGE_SORT_THRESHOLD) { __row_insertion_sort(first, last, row, ncols); return __row_unique(first, last, ncols); }
    T* mid = first + (N / 2) * ncols;
    T* a_last = __merge_sort_unique_rows1(first, mid, temp, row, ncols);
    T* b_last = __merge_sort_unique_rows1(mid, last, a_last, row, ncols);
    return __row_set_union(temp, a_last, a_last, b_last, first, ncols);
}
template <typename T>
T* __merge_sort_unique_rows1(T* first, T* last, T* temp, T* row, const intptr_t ncols) // saves the output to temp
{
    const intptr_t N = (last - first) / ncols;
    if (N <= MERGE_SORT_THRESHOLD) { __row_insertion_sort(first, last, row, ncols); return __row_unique_copy(first, last, temp, ncols); }
    T* mid = first + (N / 2) * ncols;
    T* a_last = __merge_sort_unique_rows2(first, mid, temp, row, ncols);
    T* b_last = __merge_sort_unique_rows2(mid, last, temp, row, ncols);
    return __row_set_union(first, a_last, mid, b_last, temp, ncols);
}
template <typename T>
intptr_t __merge_sort_unique_rows(T* first, T* last, const intptr_t ncols)
{
    const intptr_t N = (last - first) / ncols;
    T* row = (T*)malloc(ncols*sizeof(T));
    if (!row) { free(row); throw std::bad_alloc(); }
    if (N <= MERGE_SORT_THRESHOLD) { if (N != 0) { free(row); return 0; } __row_insertion_sort(first, last, row, ncols); free(row); return (__row_unique(first, last, ncols)-first)/ncols; }
    T* temp = (T*)malloc((N+1)/2*sizeof(T)*ncols);
    if (!temp) { free(row); throw std::bad_alloc(); }
    T* mid = first + (N/2)*ncols;
    T* b_last = __merge_sort_unique_rows1(mid, last, temp, row, ncols); // second half goes into temporary
    T* a_last = __merge_sort_unique_rows1(first, mid, mid, row, ncols); // first half goes into second half
    last = __row_set_union(mid, a_last, temp, b_last, first, ncols); // merge second half and temporary into first half (and eventually second half)
    free(temp);
    free(row);
    return (last-first)/ncols;
}

// This function makes the above usable in Cython and chooses a specialization based on the number of columns (currently 1-4 columns are specialized)
template <typename T> inline intptr_t merge_sort_unique_rows(T* first, T* last, const intptr_t ncols)
{
    switch (ncols)
    {
    case 1: return merge_sort_unique(first, last);
    case 2: return __merge_sort_unique_rows<T,2>(first, last);
    case 3: return __merge_sort_unique_rows<T,3>(first, last);
    case 4: return __merge_sort_unique_rows<T,4>(first, last);
    default: return __merge_sort_unique_rows(first, last, ncols);
    }
}


//////////////////// Binary Search for Rows ////////////////////
template <typename T, intptr_t ncols>
inline uintptr_t __row_lower_bound(const T* first, const T* last, const T* val)
{
    typedef row_ops<T,ncols> ro;
    const T *itr = first, *mid;
    intptr_t count = (last-first)/ncols, step;
    while (count > 0)
    {
        step = count/2;
        mid = itr+step*ncols;
        if (ro::lt(mid, val))
        {
            itr = mid+ncols;
            count -= step+1;
        }
        else { count = step; }
    }
    return (itr-first)/ncols;
}
template <typename T, intptr_t ncols>
inline uintptr_t __row_upper_bound(const T* first, const T* last, const T* val)
{
    typedef row_ops<T,ncols> ro;
    const T *itr = first, *mid;
    intptr_t count = (last-first)/ncols, step;
    while (count > 0)
    {
        step = count/2;
        mid = itr+step*ncols;
        if (!ro::lt(val, mid))
        {
            itr = mid+ncols;
            count -= step+1;
        }
        else { count = step; }
    }
    return (itr-first)/ncols;
}
template <typename T>
inline uintptr_t __row_lower_bound(const T* first, const T* last, const T* val, const intptr_t ncols)
{
    typedef row_ops<T> ro;
    const T *itr = first, *mid;
    intptr_t count = (last-first)/ncols, step;
    while (count > 0)
    {
        step = count/2;
        mid = itr+step*ncols;
        if (ro::lt(mid, val, ncols))
        {
            itr = mid+ncols;
            count -= step+1;
        }
        else { count = step; }
    }
    return (itr-first)/ncols;
}
template <typename T>
inline uintptr_t __row_upper_bound(const T* first, const T* last, const T* val, const intptr_t ncols)
{
    typedef row_ops<T> ro;
    const T *itr = first, *mid;
    intptr_t count = (last-first)/ncols, step;
    while (count > 0)
    {
        step = count/2;
        mid = itr+step*ncols;
        if (!ro::lt(val, mid, ncols))
        {
            itr = mid+ncols;
            count -= step+1;
        }
        else { count = step; }
    }
    return (itr-first)/ncols;
}
template <typename T>
inline void __row_lower_bounds(const T* sorted_first, const T* sorted_last,
                               const T* first, const T* last, uintptr_t* out, const intptr_t ncols)
{
    for (; first != last; first += ncols, ++out)
    {
        *out = __row_lower_bound(sorted_first, sorted_last, first, ncols);
    }
}
template <typename T>
inline void __row_upper_bounds(const T* sorted_first, const T* sorted_last,
                               const T* first, const T* last, uintptr_t* out, const intptr_t ncols)
{
    for (; first != last; first += ncols, ++out)
    {
        *out = __row_upper_bound(sorted_first, sorted_last, first, ncols);
    }
}
template <typename T, intptr_t ncols>
inline void __row_lower_bounds(const T* sorted_first, const T* sorted_last,
                               const T* first, const T* last, uintptr_t* out)
{
    for (; first != last; first += ncols, ++out)
    {
        *out = __row_lower_bound<T,ncols>(sorted_first, sorted_last, first);
    }
}
template <typename T, intptr_t ncols>
inline void __row_upper_bounds(const T* sorted_first, const T* sorted_last,
                               const T* first, const T* last, uintptr_t* out)
{
    for (; first != last; first += ncols, ++out)
    {
        *out = __row_upper_bound<T,ncols>(sorted_first, sorted_last, first);
    }
}


// These functions make the above usable in Cython and choose a specialization based on the number of columns (currently 1-4 columns are specialized)
template <typename T>
inline void row_lower_bounds(const T* sorted_first, const T* sorted_last,
                             const T* first, const T* last, uintptr_t* out, const intptr_t ncols)
{
    switch (ncols)
    {
    case 1: return __row_lower_bounds<T,1>(sorted_first, sorted_last, first, last, out);
    case 2: return __row_lower_bounds<T,2>(sorted_first, sorted_last, first, last, out);
    case 3: return __row_lower_bounds<T,3>(sorted_first, sorted_last, first, last, out);
    case 4: return __row_lower_bounds<T,4>(sorted_first, sorted_last, first, last, out);
    default: return __row_lower_bounds(sorted_first, sorted_last, first, last, out, ncols);
    }
}
template <typename T>
inline void row_upper_bounds(const T* sorted_first, const T* sorted_last,
                             const T* first, const T* last, uintptr_t* out, const intptr_t ncols)
{
    switch (ncols)
    {
    case 1: return __row_upper_bounds<T,1>(sorted_first, sorted_last, first, last, out);
    case 2: return __row_upper_bounds<T,2>(sorted_first, sorted_last, first, last, out);
    case 3: return __row_upper_bounds<T,3>(sorted_first, sorted_last, first, last, out);
    case 4: return __row_upper_bounds<T,4>(sorted_first, sorted_last, first, last, out);
    default: return __row_upper_bounds(sorted_first, sorted_last, first, last, out, ncols);
    }
}
