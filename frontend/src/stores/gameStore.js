import { create } from "zustand";
import { getBestMove } from "../services/aiService";

const TOTAL_GOATS = 20;

// Helper to create initial board with tigers in corners
const createInitialBoard = () => {
  const board = Array(5)
    .fill(null)
    .map(() => Array(5).fill(null));
  // Place tigers in corners
  const corners = [
    [0, 0],
    [0, 4],
    [4, 0],
    [4, 4],
  ];
  corners.forEach(([x, y]) => {
    board[y][x] = { type: "TIGER" };
  });
  return board;
};

// Helper to convert board to test format
const boardToTestFormat = (board) => {
  const testBoard = [];
  for (let i = 0; i < 5; i++) {
    let row = "";
    for (let j = 0; j < 5; j++) {
      const piece = board[i][j]; // Don't swap i,j anymore - we want visual representation
      if (piece === null) {
        row += "_";
      } else if (piece.type === "TIGER") {
        row += "T";
      } else {
        row += "G";
      }
    }
    testBoard.push(row);
  }
  return testBoard;
};

const initialState = {
  board: createInitialBoard(),
  turn: "GOAT", // 'GOAT' or 'TIGER'
  goatsPlaced: 0,
  goatsCaptured: 0,
  selectedPiece: null, // {x, y} of selected piece
  possibleMoves: [], // array of {x, y} positions
  phase: "PLACEMENT", // 'PLACEMENT' or 'MOVEMENT'
  gameStatus: "PLAYING", // "PLAYING", "TIGERS_WIN", "GOATS_WIN"
  moveHistory: [],
  perspective: "GOAT",
  players: {
    goat: { type: "HUMAN", model: null, settings: null }, // type: "HUMAN", "AI", model: "random", "minimax", settings: agent settings
    tiger: { type: "HUMAN", model: null, settings: null },
  },
  timeControl: {
    initial: 600, // 10 minutes in seconds
    increment: 5, // 5 seconds increment per move
  },
  tigerTime: 10,
  goatTime: 10,
  clockRunning: false,
  isInitialized: false,
  canUndo: false,
  isAIThinking: false,
  lastMove: null, // { from?: {x,y}, to: {x,y} }
  currentAbortController: null,
  isPaused: false, // New state for pause functionality
};

// Helper function to convert grid coordinates to notation
const gridToNotation = (x, y) => {
  const col = String.fromCharCode(65 + x); // Convert 0-4 to A-E
  const row = y + 1; // Convert 0-4 to 1-5
  return `${col}${row}`;
};

export const useGameStore = create((set, get) => ({
  ...initialState,

  // Add function to cancel current request
  cancelCurrentRequest: () => {
    const state = get();
    if (state.currentAbortController) {
      state.currentAbortController.abort();
      set({ currentAbortController: null, isAIThinking: false });
    }
  },

  // Reset game
  resetGame: () => {
    // First cancel any ongoing request
    get().cancelCurrentRequest();

    set({
      board: createInitialBoard(),
      phase: "PLACEMENT",
      turn: "GOAT",
      selectedPiece: null,
      possibleMoves: [],
      goatsPlaced: 0,
      goatsCaptured: 0,
      gameStatus: "PLAYING",
      moveHistory: [],
      tigerTime: initialState.timeControl.initial,
      goatTime: initialState.timeControl.initial,
      clockRunning: false,
      isInitialized: false,
      canUndo: false,
      isAIThinking: false,
      lastMove: null,
      perspective: "GOAT",
      players: {
        goat: { type: "HUMAN", model: null, settings: null },
        tiger: { type: "HUMAN", model: null, settings: null },
      },
      timeControl: {
        initial: 600,
        increment: 5,
      },
      currentAbortController: null,
      isPaused: false, // Reset paused state
    });
  },

  // Toggle pause state
  togglePause: () => {
    const state = get();

    if (state.gameStatus !== "PLAYING") return;

    const newPaused = !state.isPaused;

    // If pausing, cancel any AI move in progress
    if (newPaused && state.isAIThinking) {
      get().cancelCurrentRequest();
    }

    set({
      isPaused: newPaused,
      // When pausing, stop the clock
      clockRunning: newPaused ? false : state.clockRunning,
    });

    // If resuming, restart the clock if needed
    if (!newPaused && state.gameStatus === "PLAYING" && !state.clockRunning) {
      get().startClock();
    }
  },

  // Update settings mid-game
  updateGameSettings: (settings) => {
    const state = get();

    if (state.gameStatus !== "PLAYING" || !state.isPaused) return;

    // Determine new perspective if player types changed
    let newPerspective = state.perspective;

    // If humans and AI switch roles, update the perspective accordingly
    if (
      settings.players.goat.type !== "HUMAN" &&
      settings.players.tiger.type === "HUMAN"
    ) {
      newPerspective = "TIGER";
    } else if (
      settings.players.goat.type === "HUMAN" &&
      settings.players.tiger.type !== "HUMAN"
    ) {
      newPerspective = "GOAT";
    }

    // Update the settings
    set({
      players: settings.players,
      timeControl: settings.timeControl,
      perspective: newPerspective,
    });

    // Toggle pause state to resume the game
    setTimeout(() => {
      get().togglePause();
    }, 100);
  },

  // Select a piece
  selectPiece: (x, y) => {
    console.log("selectPiece called:", x, y);
    const state = get();

    // Don't allow interactions when paused
    if (state.isPaused) return;

    const piece = state.board[y][x];

    console.log("Current state:", {
      turn: state.turn,
      phase: state.phase,
      piece: piece,
    });

    // During PLACEMENT phase, clicking empty space should place a goat
    if (state.phase === "PLACEMENT" && state.turn === "GOAT" && !piece) {
      get().makeMove(x, y);
      return;
    }

    // If we have a selected piece and click on a valid move location, make the move
    if (
      state.selectedPiece &&
      state.possibleMoves.some((move) => move.x === x && move.y === y)
    ) {
      get().makeMove(x, y);
      return;
    }

    // If clicking a new valid piece for the current turn, select it
    // Don't show moves for goats during placement phase
    if (piece && piece.type === state.turn) {
      if (state.phase === "PLACEMENT" && piece.type === "GOAT") {
        set({ selectedPiece: null, possibleMoves: [] });
      } else {
        const possibleMoves = getPossibleMoves(x, y, state.board);
        console.log("Possible moves:", possibleMoves);
        set({ selectedPiece: { x, y }, possibleMoves });
      }
      return;
    }

    // Clear selection if:
    // 1. Clicking anywhere when a piece is selected and it's not a valid move
    // 2. Clicking a piece that's not the current player's turn
    if (
      (state.selectedPiece &&
        !state.possibleMoves.some((move) => move.x === x && move.y === y)) ||
      (piece && piece.type !== state.turn)
    ) {
      set({ selectedPiece: null, possibleMoves: [] });
      return;
    }
  },

  // Make a move
  makeMove: async (toX, toY) => {
    const state = get();

    // Don't allow moves if game is paused
    if (state.isPaused) return false;

    // Only check game status, not isAIThinking
    if (state.gameStatus !== "PLAYING") return false;

    const notation = gridToNotation(toX, toY);
    let moveNotation = "";

    // Placing a new goat
    if (state.phase === "PLACEMENT" && state.turn === "GOAT") {
      if (isValidPlacement(toX, toY, state.board)) {
        moveNotation = `G${notation}`;
        const newBoard = [...state.board.map((row) => [...row])];
        newBoard[toY][toX] = { type: "GOAT" };
        const newGoatsPlaced = state.goatsPlaced + 1;

        await new Promise((resolve) => {
          set((state) => {
            const newState = {
              ...state,
              board: newBoard,
              goatsPlaced: newGoatsPlaced,
              turn: "TIGER",
              selectedPiece: null,
              possibleMoves: [],
              phase: newGoatsPlaced >= TOTAL_GOATS ? "MOVEMENT" : "PLACEMENT",
              moveHistory: [...state.moveHistory, moveNotation],
              goatTime: state.goatTime + state.timeControl.increment,
              canUndo: true,
              lastMove: { to: { x: toX, y: toY } },
            };

            // Log board state in test format
            console.log("Current board state in test format:");
            console.log(JSON.stringify(boardToTestFormat(newBoard), null, 2));
            console.log(`Phase: ${newState.phase}`);
            console.log(`Turn: ${newState.turn}`);
            console.log(`Goats placed: ${newState.goatsPlaced}`);
            console.log(`Goats captured: ${state.goatsCaptured}`);

            resolve();
            return newState;
          });
        });

        // Ensure state is updated before checking game end
        await new Promise((resolve) => setTimeout(resolve, 50));
        await get().checkGameEnd();
        return true;
      }
      return false;
    }

    // Moving pieces (both tiger and goat)
    if (state.selectedPiece) {
      const fromNotation = gridToNotation(
        state.selectedPiece.x,
        state.selectedPiece.y
      );
      const pieceType =
        state.board[state.selectedPiece.y][state.selectedPiece.x].type;
      moveNotation = `${
        pieceType === "TIGER" ? "T" : "G"
      }${fromNotation}${notation}`;

      // Moving a tiger (allowed in both phases)
      if (state.turn === "TIGER" && state.selectedPiece) {
        if (isValidMove(state.selectedPiece, { x: toX, y: toY }, state)) {
          const newBoard = [...state.board.map((row) => [...row])];
          const { x: fromX, y: fromY } = state.selectedPiece;

          // Move the tiger piece
          newBoard[toY][toX] = newBoard[fromY][fromX];
          newBoard[fromY][fromX] = null;

          // Check if this is a capture move
          const move = state.possibleMoves.find(
            (m) => m.x === toX && m.y === toY
          );

          if (move?.type === "capture") {
            // Remove the captured goat
            newBoard[move.capturedPiece.y][move.capturedPiece.x] = null;
          }

          await new Promise((resolve) => {
            set((state) => {
              const newState = {
                ...state,
                board: newBoard,
                turn: "GOAT",
                selectedPiece: null,
                possibleMoves: [],
                goatsCaptured:
                  move?.type === "capture"
                    ? state.goatsCaptured + 1
                    : state.goatsCaptured,
                moveHistory: [...state.moveHistory, moveNotation],
                tigerTime: state.tigerTime + state.timeControl.increment,
                canUndo: true,
                lastMove: {
                  from: { x: state.selectedPiece.x, y: state.selectedPiece.y },
                  to: { x: toX, y: toY },
                },
                gameStatus:
                  move?.type === "capture" && state.goatsCaptured + 1 >= 5
                    ? "TIGERS_WIN"
                    : state.gameStatus,
              };

              // Log board state in test format
              console.log("Current board state in test format:");
              console.log(JSON.stringify(boardToTestFormat(newBoard), null, 2));
              console.log(`Phase: ${newState.phase}`);
              console.log(`Turn: ${newState.turn}`);
              console.log(`Goats placed: ${newState.goatsPlaced}`);
              console.log(`Goats captured: ${newState.goatsCaptured}`);

              resolve();
              return newState;
            });
          });

          // Ensure state is updated before checking game end
          await new Promise((resolve) => setTimeout(resolve, 50));
          await get().checkGameEnd();
          return true;
        }
        return false;
      }

      // Moving a goat during movement phase
      if (
        state.phase === "MOVEMENT" &&
        state.turn === "GOAT" &&
        state.selectedPiece
      ) {
        if (isValidMove(state.selectedPiece, { x: toX, y: toY }, state)) {
          const newBoard = [...state.board.map((row) => [...row])];
          const { x: fromX, y: fromY } = state.selectedPiece;

          // Move the piece
          newBoard[toY][toX] = newBoard[fromY][fromX];
          newBoard[fromY][fromX] = null;

          await new Promise((resolve) => {
            set((state) => {
              const newState = {
                ...state,
                board: newBoard,
                turn: "TIGER",
                selectedPiece: null,
                possibleMoves: [],
                moveHistory: [...state.moveHistory, moveNotation],
                goatTime: state.goatTime + state.timeControl.increment,
                canUndo: true,
                lastMove: {
                  from: { x: state.selectedPiece.x, y: state.selectedPiece.y },
                  to: { x: toX, y: toY },
                },
              };

              // Log board state in test format
              console.log("Current board state in test format:");
              console.log(JSON.stringify(boardToTestFormat(newBoard), null, 2));
              console.log(`Phase: ${newState.phase}`);
              console.log(`Turn: ${newState.turn}`);
              console.log(`Goats placed: ${newState.goatsPlaced}`);
              console.log(`Goats captured: ${state.goatsCaptured}`);

              resolve();
              return newState;
            });
          });

          // Ensure state is updated before checking game end
          await new Promise((resolve) => setTimeout(resolve, 50));
          await get().checkGameEnd();
          return true;
        }
        return false;
      }
    }
    return false;
  },

  getRemainingGoats: () => TOTAL_GOATS - get().goatsPlaced,

  // Add function to check for game end conditions
  checkGameEnd: async () => {
    const state = get();

    // Check if tigers won (5 goats captured)
    if (state.goatsCaptured >= 5) {
      await new Promise((resolve) => {
        set((state) => {
          const newState = { ...state, gameStatus: "TIGERS_WIN" };
          resolve();
          return newState;
        });
      });
      return;
    }

    // Check if tigers have no legal moves (works in both phases)
    let tigerHasMove = false;
    // Check each tiger position
    for (let y = 0; y < state.board.length; y++) {
      for (let x = 0; x < state.board[y].length; x++) {
        if (state.board[y][x]?.type === "TIGER") {
          const moves = getPossibleMoves(x, y, state.board);
          if (moves.length > 0) {
            tigerHasMove = true;
            break;
          }
        }
      }
      if (tigerHasMove) break;
    }

    if (!tigerHasMove) {
      await new Promise((resolve) => {
        set((state) => {
          const newState = { ...state, gameStatus: "GOATS_WIN" };
          resolve();
          return newState;
        });
      });
      return;
    }
  },

  // Start the clock
  startClock: () => {
    // Don't start the clock if game is paused
    const state = get();
    if (state.isPaused) return () => {};

    set({ clockRunning: true });
    const intervalId = setInterval(() => {
      const state = get();
      if (
        !state.clockRunning ||
        state.gameStatus !== "PLAYING" ||
        state.isPaused
      ) {
        clearInterval(intervalId);
        return;
      }

      const timeKey = state.turn === "TIGER" ? "tigerTime" : "goatTime";
      const newTime = state[timeKey] - 1;

      if (newTime <= 0) {
        // Player lost on time
        set({
          clockRunning: false,
          gameStatus: state.turn === "TIGER" ? "GOATS_WIN" : "TIGERS_WIN",
          [timeKey]: 0,
        });
        clearInterval(intervalId);
      } else {
        set({ [timeKey]: newTime });
      }
    }, 1000);

    return () => clearInterval(intervalId);
  },

  // Add time increment after a move
  addIncrement: () => {
    const state = get();
    const timeKey = state.turn === "TIGER" ? "tigerTime" : "goatTime";
    set({
      [timeKey]: state[timeKey] + state.timeControl.increment,
    });
  },

  setGameSettings: (settings) => {
    // Determine perspective based on human player
    let perspective = "GOAT"; // default
    if (
      settings.players.goat.type !== "HUMAN" &&
      settings.players.tiger.type === "HUMAN"
    ) {
      perspective = "TIGER";
    } else if (
      settings.players.goat.type === "HUMAN" &&
      settings.players.tiger.type !== "HUMAN"
    ) {
      perspective = "GOAT";
    }

    set({
      players: settings.players,
      timeControl: settings.timeControl,
      tigerTime: settings.timeControl.initial,
      goatTime: settings.timeControl.initial,
      perspective: perspective,
      isInitialized: true,
    });
  },

  undoMoves: () => {
    // First cancel any ongoing request
    get().cancelCurrentRequest();

    // Don't allow undo if game is paused
    const state = get();
    if (state.isPaused) return;

    if (state.moveHistory.length === 0) return;

    // Remove last move from history
    const newHistory = state.moveHistory.slice(0, -1);

    // Recreate the board state up to this point
    const newBoard = createInitialBoard();
    let goatsPlaced = 0;
    let goatsCaptured = 0;

    // Replay all moves except the last one
    newHistory.forEach((move) => {
      if (move.length === 3) {
        // Placement move
        const col = move.charCodeAt(1) - 65;
        const row = parseInt(move[2]) - 1;
        newBoard[row][col] = { type: "GOAT" };
        goatsPlaced++;
      } else {
        // Movement move
        const fromCol = move.charCodeAt(1) - 65;
        const fromRow = parseInt(move[2]) - 1;
        const toCol = move.charCodeAt(3) - 65;
        const toRow = parseInt(move[4]) - 1;

        // Check if it was a capture move
        if (Math.abs(fromCol - toCol) > 1 || Math.abs(fromRow - toRow) > 1) {
          goatsCaptured++;
        }

        newBoard[toRow][toCol] = newBoard[fromRow][fromCol];
        newBoard[fromRow][fromCol] = null;
      }
    });

    // Determine whose turn it should be based on move history length
    const newTurn = newHistory.length % 2 === 0 ? "GOAT" : "TIGER";

    set({
      board: newBoard,
      moveHistory: newHistory,
      goatsPlaced,
      goatsCaptured,
      turn: newTurn,
      selectedPiece: null,
      possibleMoves: [],
      phase: goatsPlaced < TOTAL_GOATS ? "PLACEMENT" : "MOVEMENT",
      canUndo: newHistory.length > 0,
    });
  },

  // Handle AI move
  handleAIMove: async () => {
    const state = get();
    if (state.gameStatus !== "PLAYING" || state.isPaused) return;

    const currentPlayer = state.players[state.turn.toLowerCase()];
    if (currentPlayer.type !== "AI" || !currentPlayer.model) return;

    // If already thinking, cancel the current request first
    if (state.isAIThinking) {
      get().cancelCurrentRequest();
    }

    console.log(
      "Starting AI move for",
      state.turn,
      "with model",
      currentPlayer.model
    );

    try {
      // Create new abort controller
      const abortController = new AbortController();
      set({ isAIThinking: true, currentAbortController: abortController });

      // Add a small delay to prevent rapid-fire moves and allow animations to complete
      await new Promise((resolve) => setTimeout(resolve, 300));

      console.log("Getting best move with state:", {
        board: state.board,
        phase: state.phase,
        turn: state.turn,
        model: currentPlayer.model,
        goatsPlaced: state.goatsPlaced,
        goatsCaptured: state.goatsCaptured,
        settings: currentPlayer.settings || null,
      });

      const move = await getBestMove(
        state.board,
        state.phase,
        state.turn,
        currentPlayer.model,
        {
          goatsPlaced: state.goatsPlaced,
          goatsCaptured: state.goatsCaptured,
          settings: currentPlayer.settings || null,
        },
        abortController
      );

      // Check if request was aborted
      if (!move) return;

      console.log("Received move:", move);

      // Verify we're still in the same turn and the request wasn't cancelled
      const currentState = get();
      if (currentState.currentAbortController !== abortController) {
        console.log("Request was cancelled or turn changed, aborting");
        return;
      }

      let success = false;
      if (move.type === "placement") {
        success = await get().makeMove(move.x, move.y);
        console.log("Placement move result:", success);
      } else {
        console.log("Selecting piece at", move.from.x, move.from.y);
        get().selectPiece(move.from.x, move.from.y);

        await new Promise((resolve) => setTimeout(resolve, 100));

        const afterSelectState = get();
        if (afterSelectState.selectedPiece) {
          console.log("Making move to", move.to.x, move.to.y);
          success = await get().makeMove(move.to.x, move.to.y);
          console.log("Movement move result:", success);
        }
      }

      if (!success) {
        console.log("Move was not successful, may need to retry");
      }
    } catch (error) {
      console.error("Error in handleAIMove:", error);
    } finally {
      set({ isAIThinking: false, currentAbortController: null });
      await new Promise((resolve) => setTimeout(resolve, 200));
    }
  },
}));

// Helper functions
function isValidPlacement(x, y, board) {
  return board[y][x] === null;
}

function isValidMove(from, to, state) {
  console.log("Validating move:", {
    from,
    to,
    possibleMoves: state.possibleMoves,
    matchingMove: state.possibleMoves.find(
      (move) => move.x === to.x && move.y === to.y
    ),
  });

  // Check if the move exists in possible moves
  const isInPossibleMoves = state.possibleMoves.some(
    (move) => move.x === to.x && move.y === to.y
  );

  // Check if destination is empty
  const isDestinationEmpty = state.board[to.y][to.x] === null;

  console.log("Move validation results:", {
    isInPossibleMoves,
    isDestinationEmpty,
  });

  return isInPossibleMoves && isDestinationEmpty;
}

function getPossibleMoves(x, y, board) {
  const moves = [];
  const piece = board[y][x];

  // Helper to check if a point is within board bounds
  const isInBounds = (x, y) => x >= 0 && x < 5 && y >= 0 && y < 5;

  // Helper to check if a point is on the outer layer
  const isOuterLayer = (x, y) => x === 0 || y === 0 || x === 4 || y === 4;

  // Helper to check if a point is on the second layer
  const isSecondLayer = (x, y) => x === 1 || y === 1 || x === 3 || y === 3;

  // Helper to check if a move is valid based on connectivity
  const isValidConnection = (fromX, fromY, toX, toY) => {
    // Orthogonal moves are always valid if adjacent
    if (Math.abs(fromX - toX) + Math.abs(fromY - toY) === 1) return true;

    // Diagonal moves need special handling
    if (Math.abs(fromX - toX) === 1 && Math.abs(fromY - toY) === 1) {
      // No diagonal moves for second and fourth nodes on outer edges
      if (isOuterLayer(fromX, fromY)) {
        const isSecondOrFourthNode =
          ((fromX === 0 || fromX === 4) && (fromY === 1 || fromY === 3)) ||
          ((fromY === 0 || fromY === 4) && (fromX === 1 || fromX === 3));
        if (isSecondOrFourthNode) return false;
      }

      // No diagonal moves for middle nodes in second layer
      if (isSecondLayer(fromX, fromY)) {
        const isMiddleNode =
          (fromX === 1 && fromY === 2) ||
          (fromX === 2 && fromY === 1) ||
          (fromX === 2 && fromY === 3) ||
          (fromX === 3 && fromY === 2);
        if (isMiddleNode) return false;
      }
      return true;
    }
    return false;
  };

  // Check regular moves
  for (let dx = -1; dx <= 1; dx++) {
    for (let dy = -1; dy <= 1; dy++) {
      if (dx === 0 && dy === 0) continue;

      const newX = x + dx;
      const newY = y + dy;

      if (
        isInBounds(newX, newY) &&
        board[newY][newX] === null &&
        isValidConnection(x, y, newX, newY)
      ) {
        moves.push({ x: newX, y: newY, type: "regular" });
      }
    }
  }

  // Add capture moves for tigers
  if (piece?.type === "TIGER") {
    // Check in all 8 directions
    const directions = [
      [-1, -1],
      [-1, 0],
      [-1, 1],
      [0, -1],
      [0, 1],
      [1, -1],
      [1, 0],
      [1, 1],
    ];

    for (const [dx, dy] of directions) {
      const midX = x + dx;
      const midY = y + dy;
      const jumpX = x + dx * 2;
      const jumpY = y + dy * 2;

      // Check if the jump is valid:
      // 1. All points must be in bounds
      // 2. Middle point must have a goat
      // 3. Destination must be empty
      // 4. Must have valid connection from tiger to goat
      // 5. Must have valid connection from goat to destination
      if (
        isInBounds(jumpX, jumpY) &&
        isInBounds(midX, midY) &&
        board[midY][midX]?.type === "GOAT" &&
        board[jumpY][jumpX] === null &&
        isValidConnection(x, y, midX, midY) && // Tiger to Goat connection
        isValidConnection(midX, midY, jumpX, jumpY) // Goat to Destination connection
      ) {
        moves.push({
          x: jumpX,
          y: jumpY,
          type: "capture",
          capturedPiece: { x: midX, y: midY },
        });
      }
    }
  }

  return moves;
}

// Export the getPossibleMoves function for use in AI service
export { getPossibleMoves };
