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

const GamePiece = ({ x, y, sprite, size, isSelected, onSelect }) => {
  const [image] = useImage(sprite);
  return (
    <>
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
        listening={true}
      />
    </>
  );
};

const Board = () => {
  const containerRef = useRef(null);
  const [dimensions, setDimensions] = useState({ width: 0, height: 0 });
  const GRID_SIZE = 5;
  const PADDING_PERCENTAGE = 0.15;
  const { board, turn, selectedPiece, possibleMoves, selectPiece, makeMove } =
    useGameStore();

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

  // Calculate dimensions based on container size
  const calculateDimensions = (containerSize) => {
    const padding = containerSize * PADDING_PERCENTAGE;
    const boardSize = containerSize - padding * 2;
    const cellSize = boardSize / (GRID_SIZE - 1);
    return { padding, boardSize, cellSize };
  };

  // Generate grid points
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

  // Generate grid lines
  const generateLines = (dims) => {
    const lines = [];

    // Horizontal and Vertical lines (existing code)
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

    // Main corner diagonals (existing)
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

    // Center point
    const midPoint = GRID_SIZE >> 1;
    const midX = dims.padding + midPoint * dims.cellSize;
    const midY = dims.padding + midPoint * dims.cellSize;

    // Diamond diagonals through grid points
    // Top to center
    lines.push({
      points: [midX, dims.padding, midX, midY],
    });
    // Right to center
    lines.push({
      points: [dims.padding + dims.boardSize, midY, midX, midY],
    });
    // Bottom to center
    lines.push({
      points: [midX, dims.padding + dims.boardSize, midX, midY],
    });
    // Left to center
    lines.push({
      points: [dims.padding, midY, midX, midY],
    });

    // Additional diagonal lines forming the diamond
    // Top center to left center
    lines.push({
      points: [midX, dims.padding, dims.padding, midY],
    });
    // Left center to bottom center
    lines.push({
      points: [dims.padding, midY, midX, dims.padding + dims.boardSize],
    });
    // Bottom center to right center
    lines.push({
      points: [
        midX,
        dims.padding + dims.boardSize,
        dims.padding + dims.boardSize,
        midY,
      ],
    });
    // Right center to top center
    lines.push({
      points: [dims.padding + dims.boardSize, midY, midX, dims.padding],
    });

    return lines;
  };

  // This will eventually come from a game state manager
  const [pieces, setPieces] = useState(() => ({
    tigers: INITIAL_TIGER_POSITIONS,
    goats: getInitialGoatPositions(),
  }));

  const boardDims = dimensions.width
    ? calculateDimensions(dimensions.width)
    : null;

  // Convert grid coordinates to pixel coordinates
  const gridToPixel = (gridX, gridY, dims) => {
    const cellSize = dims.cellSize;
    const offsetX = dims.padding;
    const offsetY = dims.padding;
    return {
      x: offsetX + gridX * cellSize,
      y: offsetY + gridY * cellSize,
    };
  };

  const handleBoardClick = (gridX, gridY) => {
    // Always use selectPiece for all clicks
    // This lets the game store handle all logic for moves and selection
    selectPiece(gridX, gridY);
  };

  return (
    <div ref={containerRef} className="w-full h-full bg-gray-800 rounded-lg">
      {boardDims && (
        <Stage width={dimensions.width} height={dimensions.height}>
          <Layer>
            <Group>
              {/* Grid lines - increased stroke width and made color whiter */}
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

              {/* Intersection points - made bigger */}
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
                return (
                  <Circle
                    key={`possible-${i}`}
                    x={pixelPos.x}
                    y={pixelPos.y}
                    radius={boardDims.cellSize * 0.2}
                    fill={
                      pos.type === "capture"
                        ? "rgba(255, 0, 0, 0.2)"
                        : "rgba(0, 255, 0, 0.2)"
                    }
                    stroke={
                      pos.type === "capture"
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
