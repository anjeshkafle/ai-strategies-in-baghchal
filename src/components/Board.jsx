import React, { useRef, useEffect, useState } from "react";
import {
  Stage,
  Layer,
  Line,
  Circle,
  Group,
  Text,
  Image,
  Rect,
} from "react-konva";
import useImage from "use-image";
import spriteGoat from "../assets/sprite_goat.png";
import spriteTiger from "../assets/sprite_tiger.png";
import { useGameStore } from "../stores/gameStore";
import Konva from "konva";

const BOARD_SIZE = 5; // 5x5 grid
const INITIAL_TIGER_POSITIONS = [
  { x: 0, y: 0 }, // Top-left
  { x: 4, y: 0 }, // Top-right
  { x: 0, y: 4 }, // Bottom-left
  { x: 4, y: 4 }, // Bottom-right
];

// This will be replaced with actual game logic later
const getInitialGoatPositions = () => {
  // Randomly place 3 goats for visualization
  const positions = [];
  const usedPositions = new Set(
    INITIAL_TIGER_POSITIONS.map((pos) => `${pos.x},${pos.y}`)
  );

  for (let i = 0; i < 3; i++) {
    let x, y;
    do {
      x = Math.floor(Math.random() * BOARD_SIZE);
      y = Math.floor(Math.random() * BOARD_SIZE);
    } while (usedPositions.has(`${x},${y}`));

    positions.push({ x, y });
    usedPositions.add(`${x},${y}`);
  }
  return positions;
};

const GamePiece = ({
  x,
  y,
  sprite,
  size,
  isSelected,
  onSelect,
  onDragStart,
  onDragEnd,
  type,
  currentTurn,
  phase,
}) => {
  const [image] = useImage(sprite);

  // Only allow dragging if:
  // 1. It's the piece's turn AND
  // 2. Either it's a tiger OR it's a goat in movement phase
  const isDraggable =
    type === currentTurn && (type === "TIGER" || phase === "MOVEMENT");

  return (
    <>
      {/* Highlight the piece if it's selected */}
      {isSelected && (
        <Circle
          x={x}
          y={y}
          radius={size * 0.4}
          fill="rgba(255, 255, 0, 0.3)"
          stroke="yellow"
          strokeWidth={2}
        />
      )}
      <Image
        x={x - (size * 0.8) / 2}
        y={y - (size * 0.8) / 2}
        image={image}
        width={size * 0.8}
        height={size * 0.8}
        onClick={onSelect}
        onTouchStart={onSelect}
        draggable={isDraggable}
        // NEW: Automatically select piece on drag start
        onDragStart={onDragStart}
        onDragEnd={onDragEnd}
        listening={true}
      />
    </>
  );
};

const Board = () => {
  const containerRef = useRef(null);
  const [dimensions, setDimensions] = useState({ width: 0, height: 0 });
  const GRID_SIZE = 5;
  const PADDING_PERCENTAGE = 0.1;

  // Grab relevant state + actions from your store
  const {
    board,
    turn,
    selectedPiece,
    possibleMoves,
    selectPiece,
    makeMove,
    phase,
  } = useGameStore();

  useEffect(() => {
    const updateDimensions = () => {
      if (containerRef.current) {
        const container = containerRef.current;
        const size = Math.min(container.offsetWidth, container.offsetHeight);
        setDimensions({ width: size, height: size });
      }
    };

    updateDimensions();
    window.addEventListener("resize", updateDimensions);
    return () => window.removeEventListener("resize", updateDimensions);
  }, []);

  // Calculate board geometry
  const calculateDimensions = (containerSize) => {
    const padding = containerSize * PADDING_PERCENTAGE;
    const boardSize = containerSize - padding * 2;
    const cellSize = boardSize / (GRID_SIZE - 1);
    return { padding, boardSize, cellSize };
  };

  // Generate grid points for circle markers
  const generatePoints = (dims) => {
    const points = [];
    for (let row = 0; row < GRID_SIZE; row++) {
      for (let col = 0; col < GRID_SIZE; col++) {
        points.push({
          x: dims.padding + col * dims.cellSize,
          y: dims.padding + row * dims.cellSize,
        });
      }
    }
    return points;
  };

  // Generate coordinate labels
  const generateLabels = (dims) => {
    const labels = [];
    const LABEL_OFFSET = 15;

    // Column letters (A-E) at bottom
    for (let col = 0; col < GRID_SIZE; col++) {
      labels.push({
        x: dims.padding + col * dims.cellSize,
        y: dims.padding + dims.boardSize + LABEL_OFFSET,
        text: String.fromCharCode(65 + col),
        type: "column",
      });
    }

    // Row numbers (1-5) on right
    for (let row = 0; row < GRID_SIZE; row++) {
      labels.push({
        x: dims.padding + dims.boardSize + LABEL_OFFSET,
        y: dims.padding + row * dims.cellSize,
        text: `${row + 1}`,
        type: "row",
      });
    }

    return labels;
  };

  // Generate lines (square + diagonal lines)
  const generateLines = (dims) => {
    const lines = [];

    // Horizontal lines
    for (let row = 0; row < GRID_SIZE; row++) {
      lines.push({
        points: [
          dims.padding,
          dims.padding + row * dims.cellSize,
          dims.padding + dims.boardSize,
          dims.padding + row * dims.cellSize,
        ],
      });
    }

    // Vertical lines
    for (let col = 0; col < GRID_SIZE; col++) {
      lines.push({
        points: [
          dims.padding + col * dims.cellSize,
          dims.padding,
          dims.padding + col * dims.cellSize,
          dims.padding + dims.boardSize,
        ],
      });
    }

    // Corner-to-corner diagonals
    lines.push({
      points: [
        dims.padding,
        dims.padding,
        dims.padding + dims.boardSize,
        dims.padding + dims.boardSize,
      ],
    });
    lines.push({
      points: [
        dims.padding + dims.boardSize,
        dims.padding,
        dims.padding,
        dims.padding + dims.boardSize,
      ],
    });

    // Main diamond lines from edges to center
    const midPoint = GRID_SIZE >> 1;
    const midX = dims.padding + midPoint * dims.cellSize;
    const midY = dims.padding + midPoint * dims.cellSize;

    // Edges to center
    lines.push({
      points: [midX, dims.padding, midX, midY],
    });
    lines.push({
      points: [dims.padding + dims.boardSize, midY, midX, midY],
    });
    lines.push({
      points: [midX, dims.padding + dims.boardSize, midX, midY],
    });
    lines.push({
      points: [dims.padding, midY, midX, midY],
    });

    // Additional diamond diagonals
    lines.push({
      points: [midX, dims.padding, dims.padding, midY],
    });
    lines.push({
      points: [dims.padding, midY, midX, dims.padding + dims.boardSize],
    });
    lines.push({
      points: [
        midX,
        dims.padding + dims.boardSize,
        dims.padding + dims.boardSize,
        midY,
      ],
    });
    lines.push({
      points: [dims.padding + dims.boardSize, midY, midX, dims.padding],
    });

    return lines;
  };

  // This will eventually come from the game store or logic
  const [pieces, setPieces] = useState(() => ({
    tigers: INITIAL_TIGER_POSITIONS,
    goats: getInitialGoatPositions(),
  }));

  const boardDims = dimensions.width
    ? calculateDimensions(dimensions.width)
    : null;

  // Convert grid coordinates to pixel coordinates
  const gridToPixel = (gridX, gridY, dims) => {
    const { cellSize, padding } = dims;
    return {
      x: padding + gridX * cellSize,
      y: padding + gridY * cellSize,
    };
  };

  // Handle board clicks (select or move)
  const handleBoardClick = (gridX, gridY) => {
    selectPiece(gridX, gridY);
  };

  // NEW: onDragStart -> select the piece immediately
  const handleDragStart = (gridX, gridY) => {
    selectPiece(gridX, gridY);
  };

  const handleDragEnd = (gridX, gridY, event) => {
    const stage = event.target.getStage();
    const pos = stage.getPointerPosition();

    if (!boardDims) return;

    const { cellSize, padding } = boardDims;
    const newGridX = Math.round((pos.x - padding) / cellSize);
    const newGridY = Math.round((pos.y - padding) / cellSize);

    // Ensure within board bounds
    if (
      newGridX >= 0 &&
      newGridX < GRID_SIZE &&
      newGridY >= 0 &&
      newGridY < GRID_SIZE &&
      (newGridX !== gridX || newGridY !== gridY)
    ) {
      // Because onDragStart selected the piece, possibleMoves is now populated
      const isValidMove = possibleMoves.some(
        (move) => move.x === newGridX && move.y === newGridY
      );

      if (isValidMove) {
        // Animate to new position
        const newPos = gridToPixel(newGridX, newGridY, boardDims);
        event.target.to({
          x: newPos.x - (cellSize * 0.6 * 0.8) / 2,
          y: newPos.y - (cellSize * 0.6 * 0.8) / 2,
          duration: 0.1,
          onFinish: () => {
            makeMove(newGridX, newGridY);
          },
        });
        return;
      }
    }

    // Otherwise, snap back
    const originalPos = gridToPixel(gridX, gridY, boardDims);
    event.target.to({
      x: originalPos.x - (cellSize * 0.6 * 0.8) / 2,
      y: originalPos.y - (cellSize * 0.6 * 0.8) / 2,
      duration: 0.2,
      easing: Konva.Easings.EaseOut,
    });
  };

  return (
    <div ref={containerRef} className="w-full h-full bg-gray-800 rounded-lg">
      {boardDims && (
        <Stage width={dimensions.width} height={dimensions.height}>
          <Layer>
            <Group>
              {/* Grid lines */}
              {generateLines(boardDims).map((line, i) => (
                <Line
                  key={i}
                  points={line.points}
                  stroke="#E5E7EB"
                  strokeWidth={2.5}
                />
              ))}

              {/* Coordinate labels */}
              {generateLabels(boardDims).map((label, i) => (
                <Text
                  key={i}
                  x={label.x}
                  y={label.y}
                  text={label.text}
                  fill="#E5E7EB"
                  fontSize={12}
                  fontFamily="monospace"
                  align="center"
                  verticalAlign="middle"
                  offsetX={6}
                  offsetY={6}
                />
              ))}

              {/* Intersection points */}
              {generatePoints(boardDims).map((point, i) => (
                <Circle
                  key={i}
                  x={point.x}
                  y={point.y}
                  radius={6}
                  fill="#E5E7EB"
                  stroke="#9CA3AF"
                  strokeWidth={2}
                />
              ))}

              {/* Possible moves indicators */}
              {possibleMoves.map((pos, i) => {
                const pixelPos = gridToPixel(pos.x, pos.y, boardDims);
                const isCapture = pos.type === "capture";
                return (
                  <Circle
                    key={`possible-${i}`}
                    x={pixelPos.x}
                    y={pixelPos.y}
                    radius={boardDims.cellSize * 0.2}
                    fill={
                      isCapture
                        ? "rgba(255, 0, 0, 0.2)"
                        : "rgba(0, 255, 0, 0.2)"
                    }
                    stroke={
                      isCapture
                        ? "rgba(255, 0, 0, 0.5)"
                        : "rgba(0, 255, 0, 0.5)"
                    }
                    strokeWidth={2}
                  />
                );
              })}

              {/* Game pieces */}
              {board.map((row, y) =>
                row.map((cell, x) => {
                  if (!cell) return null;
                  const pixelPos = gridToPixel(x, y, boardDims);
                  return (
                    <GamePiece
                      key={`piece-${x}-${y}`}
                      x={pixelPos.x}
                      y={pixelPos.y}
                      sprite={cell.type === "GOAT" ? spriteGoat : spriteTiger}
                      size={boardDims.cellSize * 0.6}
                      isSelected={
                        selectedPiece?.x === x && selectedPiece?.y === y
                      }
                      onSelect={() => handleBoardClick(x, y)}
                      onDragStart={() => handleDragStart(x, y)}
                      onDragEnd={(e) => handleDragEnd(x, y, e)}
                      type={cell.type}
                      currentTurn={turn}
                      phase={phase}
                    />
                  );
                })
              )}

              {/* Clickable areas for empty squares */}
              {board.map((row, y) =>
                row.map((cell, x) => {
                  if (cell) return null;
                  const pixelPos = gridToPixel(x, y, boardDims);
                  return (
                    <Rect
                      key={`click-${x}-${y}`}
                      x={pixelPos.x - boardDims.cellSize * 0.3}
                      y={pixelPos.y - boardDims.cellSize * 0.3}
                      width={boardDims.cellSize * 0.6}
                      height={boardDims.cellSize * 0.6}
                      fill="transparent"
                      onClick={() => handleBoardClick(x, y)}
                      onTouchStart={() => handleBoardClick(x, y)}
                    />
                  );
                })
              )}
            </Group>
          </Layer>
        </Stage>
      )}
    </div>
  );
};

export default Board;
