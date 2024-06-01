package main

import (
	"fmt"
)

func main() {
	fmt.Print(findKthLargest([]int{3, 2, 3, 1, 2, 4, 5, 5, 6}, 4))
}
func findKthLargest(nums []int, k int) int {
	// for i, n := 0, len(nums); i < n/2; i++ {
	// 	nums[i], nums[n-1-i] = nums[n-1-i], nums[i]
	// }

	l, r := 0, len(nums)-1
	for l < r {
		for l < r && nums[r] >= nums[0] {
			r--
			// for optimization
			if nums[l+1] <= nums[0] && l+1 < r {
				l++
			}
		}
		for l < r && nums[l] <= nums[0] {
			l++
		}
		if l < r {
			nums[l], nums[r] = nums[r], nums[l]
		}
	}
	nums[r], nums[0] = nums[0], nums[r]

	if l == len(nums)-k {
		return nums[l]
	} else if l < len(nums)-k {
		return findKthLargest(nums[l+1:], k)
	} else {
		return findKthLargest(nums[:l], k-(len(nums)-l))
	}
}
