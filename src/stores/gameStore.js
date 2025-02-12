import { create } from "zustand";

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

const initialState = {
  board: createInitialBoard(),
  turn: "GOAT", // 'GOAT' or 'TIGER'
  goatsPlaced: 0,
  goatsCaptured: 0,
  selectedPiece: null, // {x, y} of selected piece
  possibleMoves: [], // array of {x, y} positions
  phase: "PLACEMENT", // 'PLACEMENT' or 'MOVEMENT'
  gameStatus: "PLAYING", // "PLAYING", "TIGERS_WIN", "GOATS_WIN"
  moveHistory: [], // Add this line
  perspective: "GOAT", // Add this line
  players: {
    goat: "HUMAN", // "HUMAN" or "AI"
    tiger: "HUMAN", // "HUMAN" or "AI"
  },
  timeControl: {
    initial: 600, // 10 minutes in seconds
    increment: 5, // 5 seconds increment per move
  },
  tigerTime: 10,
  goatTime: 10,
  clockRunning: false,
  isInitialized: false, // Add this line
  canUndo: false,
};

// Helper function to convert grid coordinates to notation
const gridToNotation = (x, y) => {
  const col = String.fromCharCode(65 + x); // Convert 0-4 to A-E
  const row = y + 1; // Convert 0-4 to 1-5
  return `${col}${row}`;
};

export const useGameStore = create((set, get) => ({
  ...initialState,

  // Reset game
  resetGame: () => {
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
      isInitialized: false, // Reset this when going back to main menu
    });
  },

  // Select a piece
  selectPiece: (x, y) => {
    console.log("selectPiece called:", x, y);
    const state = get();
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
  makeMove: (toX, toY) => {
    const state = get();
    const notation = gridToNotation(toX, toY);
    let moveNotation = "";

    // Placing a new goat
    if (state.phase === "PLACEMENT" && state.turn === "GOAT") {
      if (isValidPlacement(toX, toY, state.board)) {
        moveNotation = `G${notation}`;
        const newBoard = [...state.board.map((row) => [...row])];
        newBoard[toY][toX] = { type: "GOAT" };
        const newGoatsPlaced = state.goatsPlaced + 1;
        set((state) => ({
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
        }));
      }
      return;
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
        console.log("Attempting tiger movement");
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
            const newGoatsCaptured = state.goatsCaptured + 1;
            console.log("Goats captured:", newGoatsCaptured); // Debug log

            set((state) => {
              const newState = {
                ...state,
                board: newBoard,
                turn: "GOAT",
                selectedPiece: null,
                possibleMoves: [],
                goatsCaptured: newGoatsCaptured,
                moveHistory: [...state.moveHistory, moveNotation],
                tigerTime: state.tigerTime + state.timeControl.increment,
                canUndo: true,
              };

              // Check if tigers won (5 goats captured)
              if (newGoatsCaptured >= 5) {
                console.log("Tigers should win - setting game status"); // Debug log
                newState.gameStatus = "TIGERS_WIN";
              }
              console.log("New game status:", newState.gameStatus); // Debug log

              return newState;
            });
          } else {
            set((state) => ({
              ...state,
              board: newBoard,
              turn: "GOAT",
              selectedPiece: null,
              possibleMoves: [],
              moveHistory: [...state.moveHistory, moveNotation],
              tigerTime: state.tigerTime + state.timeControl.increment,
              canUndo: true,
            }));
          }
        }
        return;
      }

      // Moving a goat during movement phase
      if (
        state.phase === "MOVEMENT" &&
        state.turn === "GOAT" &&
        state.selectedPiece
      ) {
        console.log("Attempting piece movement", {
          from: state.selectedPiece,
          to: { x: toX, y: toY },
          validMoves: state.possibleMoves,
          isValidMove: isValidMove(
            state.selectedPiece,
            { x: toX, y: toY },
            state
          ),
        });

        if (isValidMove(state.selectedPiece, { x: toX, y: toY }, state)) {
          const newBoard = [...state.board.map((row) => [...row])];
          const { x: fromX, y: fromY } = state.selectedPiece;
          console.log("Moving piece", {
            from: { x: fromX, y: fromY },
            to: { x: toX, y: toY },
            piece: newBoard[fromY][fromX],
          });

          // Move the piece
          newBoard[toY][toX] = newBoard[fromY][fromX];
          newBoard[fromY][fromX] = null;

          const move = state.possibleMoves.find(
            (move) => move.x === toX && move.y === toY
          );

          if (move?.type === "capture") {
            // Remove the captured goat
            newBoard[move.capturedPiece.y][move.capturedPiece.x] = null;
            set((state) => ({
              ...state,
              board: newBoard,
              turn: "GOAT",
              selectedPiece: null,
              possibleMoves: [],
              goatsCaptured: state.goatsCaptured + 1,
              moveHistory: [...state.moveHistory, moveNotation],
              goatTime: state.goatTime + state.timeControl.increment,
              canUndo: true,
            }));
          } else {
            set((state) => ({
              ...state,
              board: newBoard,
              turn: state.turn === "GOAT" ? "TIGER" : "GOAT",
              selectedPiece: null,
              possibleMoves: [],
              moveHistory: [...state.moveHistory, moveNotation],
              goatTime: state.goatTime + state.timeControl.increment,
              canUndo: true,
            }));
          }
        } else {
          console.log("Invalid move - Failed validation", {
            possibleMoves: state.possibleMoves,
            attemptedMove: { x: toX, y: toY },
          });
        }
      }
    }
    // After successful moves, add:
    get().checkGameEnd();
  },

  getRemainingGoats: () => TOTAL_GOATS - get().goatsPlaced,

  // Add function to check for game end conditions
  checkGameEnd: () => {
    const state = get();

    // Check if tigers won (5 goats captured)
    if (state.goatsCaptured >= 5) {
      set({ gameStatus: "TIGERS_WIN" });
      return;
    }

    // Check if goats won (tigers have no legal moves)
    if (state.turn === "TIGER") {
      let tigerHasMove = false;
      // Check each tiger position
      for (let y = 0; y < state.board.length; y++) {
        for (let x = 0; x < state.board[y].length; x++) {
          // CORRECTED: Check for tiger type in the cell object
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
        set({ gameStatus: "GOATS_WIN" });
      }
    }
  },

  // Start the clock
  startClock: () => {
    set({ clockRunning: true });
    const intervalId = setInterval(() => {
      const state = get();
      if (!state.clockRunning || state.gameStatus !== "PLAYING") {
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
    set({
      players: settings.players,
      timeControl: settings.timeControl,
      tigerTime: settings.timeControl.initial,
      goatTime: settings.timeControl.initial,
      isInitialized: true, // Set this when game settings are configured
    });
  },

  undoMoves: () => {
    const state = get();
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
