package main

import (
	"fmt"
	"strings"
)

func main() {
	fmt.Println(wordBreak("abc", []string{"a", "bb", "c"}))

}
func wordBreak(s string, wordDict []string) bool {
	dp := make([]bool, len(s)+1)
	dp[0] = true
	for i := 1; i <= len(s); i++ {
		for _, word := range wordDict {
			if strings.HasSuffix(s[:i], word) {
				dp[i] = dp[i] || dp[i-len(word)]
			}
		}
	}

	return dp[len(s)]
}
