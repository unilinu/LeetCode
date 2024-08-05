package main

import (
	"fmt"
)

func main() {
	fmt.Println(dailyTemperatures([]int{1, 2, 2, 4, 3, 5}))

}

func dailyTemperatures(temperatures []int) []int {
	n := len(temperatures)
	res := make([]int, n)
	stack := make([]int, 0, n/2)
	stack = append(stack, 0)
	for i := 1; i < n; i++ {
		top := len(stack) - 1
		for ; top >= 0 && temperatures[i] > temperatures[stack[top]]; top-- {
			res[stack[top]] = i - stack[top]
		}
		stack = stack[:top+1]
		stack = append(stack, i)
	}

	// for _, i := range stack {
	// 	res[i] = 0
	// }

	return res
}
