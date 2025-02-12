// Mock delay function
const delay = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

// Helper to get random element from array
const getRandomElement = (array) => {
  return array[Math.floor(Math.random() * array.length)];
};

// Get best move based on the current state and AI model
export const getBestMove = async (
  boardState,
  phase,
  agent,
  model = "random"
) => {
  // Check if the player is AI (anything not "HUMAN" is considered AI)
  const isAI = agent?.toLowerCase() !== "human";
  if (!isAI) return null;

  // Random delay between 1-10 seconds (1000-10000ms)
  const randomDelay = Math.floor(Math.random() * 9000) + 1000;
  await delay(randomDelay);

  // Get the piece type based on the agent
  const pieceType = agent.toUpperCase();

  // For now, just return a random legal move from all possible moves
  const allPossibleMoves = getAllPossibleMoves(boardState, phase, pieceType);
  return getRandomElement(allPossibleMoves);
};

// Helper to get all possible moves for the current state
const getAllPossibleMoves = (board, phase, pieceType) => {
  const moves = [];

  if (phase === "PLACEMENT" && pieceType === "GOAT") {
    // Get all empty spaces for goat placement
    for (let y = 0; y < board.length; y++) {
      for (let x = 0; x < board[y].length; x++) {
        if (!board[y][x]) {
          moves.push({ type: "placement", x, y });
        }
      }
    }
  } else {
    // Get all possible moves for each piece
    for (let y = 0; y < board.length; y++) {
      for (let x = 0; x < board[y].length; x++) {
        if (board[y][x]?.type === pieceType) {
          const pieceMoves = getPossibleMoves(x, y, board);
          pieceMoves.forEach((move) => {
            moves.push({
              type: "movement",
              from: { x, y },
              to: { x: move.x, y: move.y },
              capture: move.type === "capture" ? move.capturedPiece : null,
            });
          });
        }
      }
    }
  }

  return moves;
};

// Import the getPossibleMoves function from gameStore
import { getPossibleMoves } from "../stores/gameStore";
