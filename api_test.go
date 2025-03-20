package nngo

import (
    "reflect"
    "testing"
)

func TestBroadcast(t *testing.T) {
    a := NewTensor[float32]([]int{2, 3})
    a.Data = []float32{1, 2, 3, 4, 5, 6}
    expectedShape := []int{1, 1, 2, 3}
    expectedStrides := []int{0, 0, 3, 1}
    t.Run("Method API", func(t *testing.T) {
        a.RequiresGrad = true
        newShape := []int{2, 3, 1, 1}
        result := a.BroadcastTo(newShape)
        // check data
        if !reflect.DeepEqual(result.Data, a.Data) {
            t.Errorf("Transpose result incorrect, got: %v, want: %v", result.Data, a.Data)
        }
        // check shape
        if !reflect.DeepEqual(result.Shape, expectedShape) {
            t.Errorf("Transpose shape incorrect, got: %v, want: %v", result.Shape, expectedShape)
        }
        // check strides
        if !reflect.DeepEqual(result.Strides, expectedStrides) {
            t.Errorf("Transpose strides incorrect, got: %v, want: %v", result.Strides, expectedStrides)
        }
    })
}

func TestTranspose(t *testing.T) {
    a := NewTensor[float32]([]int{2, 3})
    a.Data = []float32{1, 2, 3, 4, 5, 6}
    expectedShape := []int{3, 2}
    expectedStrides := []int{1, 3}
    t.Run("Function API", func(t *testing.T) {
        a.RequiresGrad = true
        newOrder := []int{1, 0}
        result := Transpose(a, newOrder)
        // check data
        if !reflect.DeepEqual(result.Data, a.Data) {
            t.Errorf("Transpose result incorrect, got: %v, want: %v", result.Data, a.Data)
        }
        // check shape
        if !reflect.DeepEqual(result.Shape, expectedShape) {
            t.Errorf("Transpose shape incorrect, got: %v, want: %v", result.Shape, expectedShape)
        }
        // check strides
        if !reflect.DeepEqual(result.Strides, expectedStrides) {
            t.Errorf("Transpose strides incorrect, got: %v, want: %v", result.Strides, expectedStrides)
        }
        // check that GradFn is not nil when RequiresGrad is true
        if result.GradFn == nil {
            t.Errorf("GradFn should not be nil when RequiresGrad is true")
        }
    })
}

func TestReshape(t *testing.T) {
    a := NewTensor[float32]([]int{2, 3})
    a.Data = []float32{1, 2, 3, 4, 5, 6}
    t.Run("Function API", func(t *testing.T) {
        a.RequiresGrad = true
        newShape := []int{3, 2}
        result := Reshape(a, newShape)
        if !reflect.DeepEqual(result.Data, a.Data) {
            t.Errorf("Add result incorrect, got: %v, want: %v", result.Data, a.Data)
        }
        // check that GradFn is not nil when RequiresGrad is true
        if result.GradFn == nil {
            t.Errorf("GradFn should not be nil when RequiresGrad is true")
        }
    })
}

func TestAdd(t *testing.T) {
    // create test tensors
    a := NewTensor[float32]([]int{2, 3})
    b := NewTensor[float32]([]int{2, 3})
    a.Data = []float32{1, 2, 3, 4, 5, 6}
    b.Data = []float32{7, 8, 9, 10, 11, 12}
    // expected result after addition
    expected := []float32{8, 10, 12, 14, 16, 18}
    // test the Add function
    t.Run("Function API", func(t *testing.T) {
        // enable grad tracking for one tensor to test grad functionality
        a.RequiresGrad = true
        // use the API function
        result := Add(a, b)
        // check the result data
        if !reflect.DeepEqual(result.Data, expected) {
            t.Errorf("Add result incorrect, got: %v, want: %v", result.Data, expected)
        }
        // check that RequiresGrad was properly set
        if !result.RequiresGrad {
            t.Errorf("RequiresGrad not properly propagated, expected true")
        }
        // check that GradFn is not nil when RequiresGrad is true
        if result.GradFn == nil {
            t.Errorf("GradFn should not be nil when RequiresGrad is true")
        }
    })

    // test the method-based API
    t.Run("Method API", func(t *testing.T) {
        // reset RequiresGrad flag
        a.RequiresGrad = false
        b.RequiresGrad = true
        // use the method API
        result := a.Add(b)
        // check the result data
        if !reflect.DeepEqual(result.Data, expected) {
            t.Errorf("Add method result incorrect, got: %v, want: %v", result.Data, expected)
        }
        // check that RequiresGrad was properly set
        if !result.RequiresGrad {
            t.Errorf("RequiresGrad not properly propagated, expected true")
        }
    })
}

func TestMul(t *testing.T) {
    // create test tensors
    a := NewTensor[float32]([]int{2, 3})
    b := NewTensor[float32]([]int{2, 3})
    a.Data = []float32{1, 2, 3, 4, 5, 6}
    b.Data = []float32{7, 8, 9, 10, 11, 12}
    // expected result after addition
    expected := []float32{7, 16, 27, 40, 55, 72}
    // test the Add function
    t.Run("Function API", func(t *testing.T) {
        // enable grad tracking for one tensor to test grad functionality
        a.RequiresGrad = true
        // use the API function
        result := Mul(a, b)
        // check the result data
        if !reflect.DeepEqual(result.Data, expected) {
            t.Errorf("Add result incorrect, got: %v, want: %v", result.Data, expected)
        }
        // check that RequiresGrad was properly set
        if !result.RequiresGrad {
            t.Errorf("RequiresGrad not properly propagated, expected true")
        }
        // check that GradFn is not nil when RequiresGrad is true
        if result.GradFn == nil {
            t.Errorf("GradFn should not be nil when RequiresGrad is true")
        }
    })

    // test the method-based API
    t.Run("Method API", func(t *testing.T) {
        // reset RequiresGrad flag
        a.RequiresGrad = false
        b.RequiresGrad = true
        // use the method API
        result := a.Mul(b)
        // check the result data
        if !reflect.DeepEqual(result.Data, expected) {
            t.Errorf("Add method result incorrect, got: %v, want: %v", result.Data, expected)
        }
        // check that RequiresGrad was properly set
        if !result.RequiresGrad {
            t.Errorf("RequiresGrad not properly propagated, expected true")
        }
    })
}
