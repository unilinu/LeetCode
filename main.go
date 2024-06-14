package main

func main() {

}
func shortestBridge(grid [][]int) (step int) {
	type seat struct {
		x, y int
	}
	q := make([]seat, 0)
	for i, row := range grid {
		for j, unit := range row {
			if unit == 1 {
				q = append(q, seat{i, j})
				break
			}
		}
		if len(q) != 0 {
			break
		}
	}

	if len(q) == 0 {
		return
	}

	dirs := []int{-1, 0, 1, 0, -1}
	m, n := len(grid), len(grid[0])

	// find island
	for i := 0; i < len(q); i++ {
		p := q[i]

		grid[p.x][p.y] = 2
		for i := 0; i < 4; i++ {
			if x, y := p.x+dirs[i], p.y+dirs[i+1]; x >= 0 && x < m &&
				y >= 0 && y < n && grid[x][y] == 1 {
				grid[x][y] = 2
				q = append(q, seat{x, y})
			}
		}
	}

	// expand island by BFS
	q2 := make([]seat, 0)
	for len(q) != 0 {
		p := q[0]
		q = q[1:]

		for i := 0; i < 4; i++ {
			if x, y := p.x+dirs[i], p.y+dirs[i+1]; x >= 0 && x < m && y >= 0 && y < n {
				if grid[x][y] == 0 {
					grid[x][y] = 2
					q2 = append(q2, seat{x, y})
				} else if grid[x][y] == 1 {
					return
				}
			}
		}

		if len(q) == 0 {
			step++
			q, q2 = q2, make([]seat, 0)
		}
	}

	return
}
