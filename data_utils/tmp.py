class Solution:
  def gameOfLife(self, board) -> None:
      """
      Do not return anything, modify board in-place instead.
      """
      turn_one = []
      turn_zero = []
      m,n = len(board),len(board[0])
      for i in range(m):
          for j in range(n):
              count_one = 0
              for k in range(i-1,i+2):
                  if k >= 0 and k < m:
                      for l in range(j-1,j+2):
                          if l >= 0 and l < n:
                              if board[k][l] == 1 and not (k==i and l==j):
                                  count_one += 1
              if count_one < 2 or count_one > 3:
                  if board[i][j] == 1: turn_zero.append([i,j])
              elif count_one == 3:
                  if board[i][j] == 0: turn_one.append([i,j])
      
      for t in turn_zero:
          board[t[0]][t[1]] = 0
      for t in turn_one:
          board[t[0]][t[1]] = 1

if __name__ == '__main__':
    board = [[0,1,0],[0,0,1],[1,1,1],[0,0,0]]
    solution = Solution()
    solution.gameOfLife(board)
    print(board)