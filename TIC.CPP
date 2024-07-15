#include <iostream>
#include <vector>

void printBoard(const std::vector<std::vector<char>> &board);
bool isMoveValid(const std::vector<std::vector<char>> &board, int row, int col);
bool checkWin(const std::vector<std::vector<char>> &board, char player);
bool checkDraw(const std::vector<std::vector<char>> &board);
bool playAgain();

int main()
{
    char currentPlayer = 'X';

    std::cout << "\n=== Welcome to Tic-Tac-Toe! ===\n"
              << std::endl;

    do
    {
        std::vector<std::vector<char>> board(3, std::vector<char>(3, ' '));
        bool gameActive = true;
        while (gameActive)
        {
            printBoard(board);

            int row, col;
            std::cout << "Player " << currentPlayer << ", enter the row (1-3) and column (1-3) numbers of your move: ";
            std::cin >> row >> col;

            if (row >= 1 && row <= 3 && col >= 1 && col <= 3)
            {
                if (isMoveValid(board, row - 1, col - 1))
                {
                    board[row - 1][col - 1] = currentPlayer;

                    if (checkWin(board, currentPlayer))
                    {
                        printBoard(board);
                        std::cout << "\nPlayer " << currentPlayer << " wins! Congratulations!\n"
                                  << std::endl;
                        gameActive = false;
                    }
                    else
                    {
                        if (checkDraw(board))
                        {
                            printBoard(board);
                            std::cout << "\nIt's a draw!\n"
                                      << std::endl;
                            gameActive = false;
                        }
                        else
                        {
                            currentPlayer = (currentPlayer == 'X') ? 'O' : 'X';
                        }
                    }
                }
                else
                {
                    std::cout << "\nInvalid move! Please try again.\n"
                              << std::endl;
                }
            }
            else
            {
                std::cout << "\nInvalid input! Row and column values must be between 1 and 3.\n"
                          << std::endl;
            }
        }

    } while (playAgain());

    std::cout << "Thank you for playing Tic-Tac-Toe!\n"
              << std::endl;

    return 0;
}
void printBoard(const std::vector<std::vector<char>> &board)
{
    std::cout << std::endl;
    for (int i = 0; i < 3; ++i)
    {
        for (int j = 0; j < 3; ++j)
        {
            std::cout << " " << board[i][j];
            if (j < 2)
                std::cout << " |";
        }
        std::cout << std::endl;
        if (i < 2)
            std::cout << "---+---+---" << std::endl;
    }
    std::cout << std::endl;
}
bool isMoveValid(const std::vector<std::vector<char>> &board, int row, int col)
{
    return (board[row][col] == ' ');
}
bool checkWin(const std::vector<std::vector<char>> &board, char player)
{
    for (int i = 0; i < 3; ++i)
    {
        if (board[i][0] == player && board[i][1] == player && board[i][2] == player)
            return true;
        if (board[0][i] == player && board[1][i] == player && board[2][i] == player)
            return true;
    }
    if (board[0][0] == player && board[1][1] == player && board[2][2] == player)
        return true;
    if (board[0][2] == player && board[1][1] == player && board[2][0] == player)
        return true;

    return false;
}
bool checkDraw(const std::vector<std::vector<char>> &board)
{
    for (const auto &row : board)
    {
        for (char cell : row)
        {
            if (cell == ' ')
                return false;
        }
    }
    return true;
}

bool playAgain()
{
    char choice;
    std::cout << "Do you want to play again? (y/n): ";
    std::cin >> choice;
    return (choice == 'y' || choice == 'Y');
}