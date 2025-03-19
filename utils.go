package nngo

import "fmt"

// CastToInts
func CastToInts(s []any) ([]int, error) {
    nS := make([]int, len(s))
    for i, dim := range s {
        nDim, ok := dim.(int)
        if !ok { return nil, fmt.Errorf("could not convert to int") }
        nS[i] = nDim
    }
    return nS, nil
}

// ComputeStrides
func ComputeStrides(shape []int) []int {
    strides := make([]int, len(shape))
    val := 1
    for i := len(shape)-1; i >= 0; i--{
        strides[i] = val
        val *= shape[i]
    }
    return strides
}

// ComputeSize
func ComputeSize(s []int) int {
    size := 1
    for _, dim := range s {
        size *= dim
    }
    return size
}

// BroadcastShapes
func BroadcastShapes(x1, x2 []int) ([]int, error) {
    // swap slices if x1 is lower dim than x2
    if len(x1) < len(x2) { x1, x2 = x2, x1 }
    // save output shape
    result := make([]int, len(x1))
    // fill in dimensions from right to left
    for i := range x1 {
        // for dimensions that x2 doesn't have, use x1's dimensions
        if i >= len(x1) - len(x2) {
            // get the corresponding dimension in x22
            x2Idx := i - (len(x1) - len(x2))
            // if dimensions match use either and, if both dimensions are 1 result is 1
            if x1[i] == x2[x2Idx] {
                result[i] = x1[i]
            // if one dimension is 1, use the other
            } else if x1[i] == 1 {
                result[i] = x2[x2Idx]
            } else if x2[x2Idx] == 1 {
                result[i] = x1[i]
            // if dimensions don't match and neither is 1, broadcasting fails
            } else {
                return nil, fmt.Errorf("cannot broadcast shapes %v and %v: incompatible dimensions at axis %d: %d vs %d", 
                    x1, x2, i, x1[i], x2[x2Idx])
            }
        } else {
            // For dimensions that b doesn't have, use a's dimensions
            result[i] = x1[i]
        }
    }
    return result, nil
}
