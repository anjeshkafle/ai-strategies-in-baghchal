import React, { useEffect } from "react";
import Board from "../components/Board";
import { useGameStore } from "../stores/gameStore";
import { useNavigate } from "react-router-dom";
import useImage from "use-image";
import spriteGoat from "../assets/sprite_goat.png";
import spriteTiger from "../assets/sprite_tiger.png";

const GameScreen = () => {
  const navigate = useNavigate();
  const {
    turn,
    phase,
    getRemainingGoats,
    goatsCaptured,
    gameStatus,
    resetGame,
    moveHistory,
    perspective,
    tigerTime,
    goatTime,
    startClock,
  } = useGameStore();
  const remainingGoats = getRemainingGoats();

  const handleNewGame = () => {
    resetGame();
  };

  const handleMainMenu = () => {
    resetGame();
    navigate("/");
  };

  const formatTime = (seconds) => {
    const minutes = Math.floor(seconds / 60);
    const remainingSeconds = seconds % 60;
    return `${minutes}:${remainingSeconds.toString().padStart(2, "0")}`;
  };

  const getTimeStatus = (seconds) => {
    if (seconds <= 10) return "critical";
    if (seconds <= 30) return "warning";
    return "normal";
  };

  const TIME_STATUS_STYLES = {
    normal: "",
    warning: "text-yellow-400",
    critical: "text-red-500",
  };

  const ACTIVE_TIME_STATUS_STYLES = {
    normal: "",
    warning: "text-yellow-400 animate-pulse",
    critical: "text-red-500 animate-[pulse_0.5s_ease-in-out_infinite]",
  };

  useEffect(() => {
    if (gameStatus === "PLAYING") {
      const cleanup = startClock();
      return cleanup;
    }
  }, [gameStatus, startClock]);

  // Helper function to render player panel
  const renderPlayerPanel = (isTopPanel) => {
    const isTiger = (perspective === "TIGER") !== isTopPanel;
    const playerType = isTiger ? "TIGER" : "GOAT";
    const sprite = isTiger ? spriteTiger : spriteGoat;
    const [spriteImage] = useImage(sprite);

    // Helper function for goats captured text
    const getCapturedText = (count) => {
      return count === 1 ? "1 GOAT CAPTURED" : `${count} GOATS CAPTURED`;
    };

    // Helper function for remaining goats text
    const getRemainingText = (count) => {
      return count === 1 ? "1 GOAT REMAINING" : `${count} GOATS REMAINING`;
    };

    return (
      <div className="flex-none p-3 border-b border-gray-700">
        {/* Player info section */}
        <div>
          <div
            className={`flex items-center justify-between ${
              turn === playerType
                ? isTiger
                  ? "text-red-400"
                  : "text-green-400"
                : "text-gray-400"
            }`}
          >
            <div className="flex items-center gap-3">
              {spriteImage && (
                <div className="relative">
                  <img
                    src={sprite}
                    alt={isTiger ? "Tiger" : "Goat"}
                    className="w-8 h-8 object-contain"
                  />
                  {turn === playerType && (
                    <div className="absolute -bottom-1.5 -right-1.5 w-2.5 h-2.5 bg-yellow-400 rounded-full animate-pulse" />
                  )}
                </div>
              )}
              <div className="flex flex-col">
                <span className="text-xs text-gray-500 uppercase tracking-wider">
                  Player
                </span>
                <span className="text-base font-extrabold tracking-wide">
                  {isTiger ? "TIGER" : "GOAT"}
                </span>
              </div>
            </div>
            <div
              className={`flex flex-col items-end ${
                turn === playerType
                  ? isTiger
                    ? "text-red-400"
                    : "text-green-400"
                  : "text-gray-400"
              }`}
            >
              <span className="text-xs text-gray-500 uppercase tracking-wider">
                Time
              </span>
              <span
                className={`text-xl font-mono font-bold ${
                  turn === playerType
                    ? ACTIVE_TIME_STATUS_STYLES[
                        getTimeStatus(isTiger ? tigerTime : goatTime)
                      ]
                    : TIME_STATUS_STYLES[
                        getTimeStatus(isTiger ? tigerTime : goatTime)
                      ]
                }`}
              >
                {formatTime(isTiger ? tigerTime : goatTime)}
              </span>
            </div>
          </div>
        </div>

        {/* Status info section with separator */}
        {gameStatus === "PLAYING" && (
          <>
            <div className="border-t border-gray-700 my-2" />
            <div className="text-gray-400 text-sm">
              <div className="flex justify-between items-center">
                {isTiger ? (
                  <span className="text-red-400 font-mono font-bold tracking-wide">
                    {getCapturedText(goatsCaptured)}
                  </span>
                ) : (
                  <span className="text-yellow-400 font-mono font-bold tracking-wide">
                    {phase === "PLACEMENT"
                      ? getRemainingText(remainingGoats)
                      : "MOVEMENT PHASE"}
                  </span>
                )}
              </div>
            </div>
          </>
        )}
      </div>
    );
  };

  return (
    <div className="fixed inset-0 bg-gray-900">
      {/* Top Navigation */}
      <div className="h-12 bg-gray-800 border-b border-gray-700">
        <div className="container mx-auto px-4 h-full flex items-center justify-between">
          <div className="text-white font-bold text-xl">Baghchal</div>
          <div className="flex items-center gap-4">
            <button className="text-gray-300 hover:text-white">Rules</button>
            <button className="text-gray-300 hover:text-white">Settings</button>
          </div>
        </div>
      </div>

      {/* Game Status Banner - Add this near the top of your layout */}
      {gameStatus !== "PLAYING" && (
        <div className="absolute inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-gray-800 p-8 rounded-lg shadow-xl text-center space-y-6">
            <h2 className="text-3xl font-bold text-yellow-400">
              {gameStatus === "TIGERS_WIN" ? "Tigers Win!" : "Goats Win!"}
            </h2>
            <p className="text-gray-300">
              {gameStatus === "TIGERS_WIN"
                ? goatsCaptured >= 5
                  ? "Tigers captured 5 goats!"
                  : "Goats ran out of time!"
                : gameStatus === "GOATS_WIN"
                ? tigerTime <= 0
                  ? "Tigers ran out of time!"
                  : "Tigers have no legal moves left!"
                : ""}
            </p>
            <div className="space-x-4">
              <button
                onClick={handleNewGame}
                className="px-6 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
              >
                New Game
              </button>
              <button
                onClick={handleMainMenu}
                className="px-6 py-2 bg-gray-600 text-white rounded-lg hover:bg-gray-700 transition-colors"
              >
                Main Menu
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Main Content */}
      <div className="mx-auto h-[calc(100vh-3rem)] flex justify-center">
        <div className="flex gap-2 p-2 items-center">
          {/* Board - Added specific dimensions */}
          <div className="bg-gray-800 rounded-lg aspect-square h-[calc(100vh-4rem)]">
            <Board />
          </div>

          {/* Right Section */}
          <div className="flex flex-col gap-2 w-72 self-center max-h-[calc(100vh-6rem)]">
            {/* Game Info Panel */}
            <div className="bg-gray-800 rounded-lg flex flex-col">
              {/* Opponent Panel */}
              {renderPlayerPanel(true)}

              {/* Move List */}
              <div className="flex-grow overflow-y-auto custom-scrollbar h-[40vh] border-b border-gray-700">
                <div className="p-2">
                  <div className="space-y-1">
                    {/* Group moves into pairs */}
                    {Array.from({
                      length: Math.ceil(moveHistory.length / 2),
                    }).map((_, i) => {
                      const goatMove = moveHistory[i * 2];
                      const tigerMove = moveHistory[i * 2 + 1];

                      // Helper to check if a move is a capture (when source and destination are 2 squares apart)
                      const isCapture = (move) => {
                        if (!move || move.length !== 5) return false;
                        const sourceCol = move.charCodeAt(1) - 65;
                        const sourceRow = parseInt(move[2]) - 1;
                        const destCol = move.charCodeAt(3) - 65;
                        const destRow = parseInt(move[4]) - 1;
                        return (
                          Math.abs(sourceCol - destCol) > 1 ||
                          Math.abs(sourceRow - destRow) > 1
                        );
                      };

                      return (
                        <div key={i} className="flex text-xs">
                          <span className="w-6 text-gray-500">{i + 1}.</span>
                          <span className="w-[45%] text-gray-300">
                            {goatMove &&
                              (goatMove.length === 3
                                ? `Goat to ${goatMove.slice(1)}`
                                : `Goat ${goatMove.slice(
                                    1,
                                    3
                                  )} to ${goatMove.slice(3)}`)}
                          </span>
                          <span
                            className={`w-[45%] ${
                              isCapture(tigerMove)
                                ? "text-red-400"
                                : "text-gray-300"
                            }`}
                          >
                            {tigerMove &&
                              `Tiger ${tigerMove.slice(
                                1,
                                3
                              )} to ${tigerMove.slice(3)}`}
                          </span>
                        </div>
                      );
                    })}
                  </div>
                </div>
              </div>

              {/* Player Panel */}
              {renderPlayerPanel(false)}
            </div>

            {/* Action Buttons */}
            <div className="bg-gray-800 rounded-lg p-2 flex gap-2">
              <button className="flex-1 px-3 py-2 bg-yellow-600 hover:bg-yellow-700 text-white text-sm font-medium rounded">
                Offer Draw
              </button>
              <button className="flex-1 px-3 py-2 bg-red-600 hover:bg-red-700 text-white text-sm font-medium rounded">
                Resign
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default GameScreen;
