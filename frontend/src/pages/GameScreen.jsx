import React, { useEffect } from "react";
import Board from "../components/Board";
import { useGameStore } from "../stores/gameStore";
import { useNavigate } from "react-router-dom";
import useImage from "use-image";
import spriteGoat from "../assets/sprite_goat.png";
import spriteTiger from "../assets/sprite_tiger.png";
import Modal from "../components/Modal";
import GameSettingsPanel from "../components/GameSettingsPanel";

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
    canUndo,
    undoMoves,
    isAIThinking,
    players,
    handleAIMove,
    isPaused,
    togglePause,
    updateGameSettings,
    timeControl,
  } = useGameStore();
  const remainingGoats = getRemainingGoats();
  const [isModalOpen, setIsModalOpen] = React.useState(false);

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
    if (gameStatus === "PLAYING" && !isPaused) {
      const cleanup = startClock();
      return cleanup;
    }
  }, [gameStatus, startClock, isPaused]);

  useEffect(() => {
    if (gameStatus === "PLAYING" && !isPaused) {
      const currentPlayer = players[turn.toLowerCase()];
      if (currentPlayer.type === "AI" && !isAIThinking) {
        // Add a small delay before triggering AI move to allow for animations
        const timeoutId = setTimeout(() => {
          handleAIMove();
        }, 300);

        return () => clearTimeout(timeoutId);
      }
    }
  }, [turn, phase, gameStatus, players, handleAIMove, isAIThinking, isPaused]);

  // Helper function to render player panel
  const renderPlayerPanel = (isTopPanel) => {
    const isTiger = (perspective === "TIGER") !== isTopPanel;
    const playerType = isTiger ? "TIGER" : "GOAT";
    const sprite = isTiger ? spriteTiger : spriteGoat;
    const [spriteImage] = useImage(sprite);

    const isWinner =
      (gameStatus === "TIGERS_WIN" && isTiger) ||
      (gameStatus === "GOATS_WIN" && !isTiger);

    // Helper function for goats captured text
    const getCapturedText = (count) => {
      return count === 1 ? "1 GOAT CAPTURED" : `${count} GOATS CAPTURED`;
    };

    // Helper function for remaining goats text
    const getRemainingText = (count) => {
      return count === 1 ? "1 GOAT REMAINING" : `${count} GOATS REMAINING`;
    };

    return (
      <div
        className={`flex-none p-3 border-b border-gray-700 ${
          gameStatus !== "PLAYING" && isWinner
            ? "bg-gradient-to-r from-yellow-500/10 to-transparent"
            : ""
        }`}
      >
        {/* Player info section */}
        <div>
          <div
            className={`flex items-center justify-between ${
              isWinner
                ? "text-yellow-400"
                : turn === playerType
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
                  {turn === playerType && !isPaused && (
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
                turn === playerType && !isPaused
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
                  turn === playerType && !isPaused
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
              {/* Add a fixed-height container for the thinking indicator */}
              <div className="h-5">
                {" "}
                {/* This creates a fixed space */}
                {isAIThinking && playerType === turn && (
                  <div className="text-yellow-400 text-sm animate-pulse">
                    Thinking...
                  </div>
                )}
              </div>
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

  const handleStopGame = () => {
    setIsModalOpen(true);
  };

  const handleConfirmStop = () => {
    setIsModalOpen(false);
    handleMainMenu();
  };

  const handleCancelStop = () => {
    setIsModalOpen(false);
  };

  const handleTogglePause = () => {
    togglePause();
  };

  const handleApplySettings = (newSettings) => {
    updateGameSettings(newSettings);
  };

  return (
    <div className="fixed inset-0 bg-gray-900">
      {/* Top Navigation */}
      <div className="h-12 bg-gray-800 border-b border-gray-700">
        <div className="container mx-auto px-4 h-full flex items-center justify-between">
          <div className="flex items-center gap-4">
            <div className="flex items-center gap-3">
              <img src={spriteGoat} alt="Goat" className="w-8 h-8" />
              <h1 className="text-2xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-yellow-400 to-yellow-600">
                Baghchal
              </h1>
              <img src={spriteTiger} alt="Tiger" className="w-8 h-8" />
            </div>
          </div>
          <div className="hidden md:flex items-center gap-4">
            <button
              onClick={handleTogglePause}
              className={`text-gray-300 hover:text-white hover:bg-gray-700 px-3 py-1.5 rounded-md transition-colors ${
                isPaused ? "bg-gray-700" : ""
              }`}
            >
              {isPaused ? "Resume" : "Pause"}
            </button>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="h-[calc(100vh-3rem)] overflow-y-auto">
        {/* Mobile Layout */}
        <div className="md:hidden flex flex-col min-h-full">
          {/* Top Player Panel */}
          <div className="bg-gray-800">{renderPlayerPanel(true)}</div>

          {/* Board */}
          <div className="flex-1 aspect-square w-full bg-gray-800 relative">
            <Board />
            {/* Pause overlay for mobile */}
            {isPaused && (
              <div className="absolute inset-0 bg-black bg-opacity-50 backdrop-blur-[1px] flex items-center justify-center">
                <div className="bg-gray-800/90 p-4 rounded-lg text-center">
                  <h3 className="text-xl font-bold text-white mb-2">
                    Game Paused
                  </h3>
                  <button
                    onClick={handleTogglePause}
                    className="px-4 py-2 bg-green-600 hover:bg-green-700 text-white rounded"
                  >
                    Resume Game
                  </button>
                </div>
              </div>
            )}
          </div>

          {/* Bottom Player Panel and Actions */}
          <div className="bg-gray-800">
            {renderPlayerPanel(false)}
            <div className="p-2 flex gap-2">
              {gameStatus === "PLAYING" ? (
                <>
                  <button
                    onClick={handleTogglePause}
                    className={`flex-1 px-3 py-2 ${
                      isPaused
                        ? "bg-green-600 hover:bg-green-700"
                        : "bg-yellow-600 hover:bg-yellow-700"
                    } text-white text-sm font-medium rounded`}
                  >
                    {isPaused ? "Resume" : "Pause"}
                  </button>
                  <button
                    onClick={handleStopGame}
                    className="flex-1 px-3 py-2 bg-red-600 hover:bg-red-700 text-white text-sm font-medium rounded"
                  >
                    Stop
                  </button>
                  <button
                    onClick={undoMoves}
                    className={`flex-1 px-3 py-2 ${
                      canUndo && !isPaused
                        ? "bg-blue-600 hover:bg-blue-700"
                        : "bg-gray-600 cursor-not-allowed"
                    } text-white text-sm font-medium rounded`}
                    disabled={!canUndo || isPaused}
                  >
                    Undo
                  </button>
                </>
              ) : (
                <button
                  onClick={handleMainMenu}
                  className="w-full px-3 py-2 bg-gray-600 hover:bg-gray-700 text-white text-sm font-medium rounded"
                >
                  Main Menu
                </button>
              )}
            </div>
          </div>
        </div>

        {/* Desktop Layout */}
        <div className="hidden md:flex gap-2 p-2 items-stretch justify-between h-full w-full">
          {/* Left Section - Game Settings Panel */}
          <div className="flex-1 self-stretch max-w-md">
            <GameSettingsPanel
              isPaused={isPaused}
              onTogglePause={handleTogglePause}
              onApplySettings={handleApplySettings}
            />
          </div>

          {/* Center and Right Sections - Combined */}
          <div className="flex gap-2 items-center">
            {/* Board - Now in the middle */}
            <div className="bg-gray-800 rounded-lg aspect-square h-[calc(100vh-4.5rem)] relative">
              <Board />
              {gameStatus !== "PLAYING" && (
                <div className="absolute inset-0 bg-black bg-opacity-50 backdrop-blur-[2px] flex items-center justify-center p-8">
                  <div className="w-full max-w-lg bg-gray-800/90 rounded-xl p-8 shadow-2xl border border-gray-700/50">
                    <div className="text-center space-y-4">
                      <h2 className="text-5xl font-bold bg-gradient-to-r from-yellow-400 to-yellow-600 bg-clip-text text-transparent">
                        {gameStatus === "TIGERS_WIN"
                          ? "Tigers Win!"
                          : "Goats Win!"}
                      </h2>
                      <p className="text-xl text-gray-300 font-medium">
                        {gameStatus === "TIGERS_WIN"
                          ? goatsCaptured >= 5
                            ? "Tigers captured 5 goats"
                            : "Goats ran out of time"
                          : tigerTime <= 0
                          ? "Tigers ran out of time"
                          : "Tigers have no legal moves"}
                      </p>
                    </div>
                  </div>
                </div>
              )}
              {/* Pause overlay for desktop */}
              {isPaused && gameStatus === "PLAYING" && (
                <div className="absolute inset-0 bg-black bg-opacity-40 backdrop-blur-[1px] flex items-center justify-center">
                  <div className="bg-gradient-to-b from-yellow-500/20 to-transparent p-6 rounded-full">
                    <div className="text-4xl font-bold text-white text-center">
                      PAUSED
                    </div>
                  </div>
                </div>
              )}
            </div>

            {/* Right Section - Player Info & Moves */}
            <div className="flex flex-col gap-2 w-72 h-[calc(100vh-4.5rem)] self-center">
              <div className="bg-gray-800 rounded-lg flex flex-col flex-grow">
                {renderPlayerPanel(true)}
                <div className="flex-grow overflow-y-auto custom-scrollbar h-[50vh] border-b border-gray-700">
                  <div className="p-2">
                    <div className="space-y-1">
                      {/* Group moves into pairs */}
                      {Array.from({
                        length: Math.ceil(moveHistory.length / 2),
                      }).map((_, i) => {
                        const goatMove = moveHistory[i * 2];
                        const tigerMove = moveHistory[i * 2 + 1];

                        // Helper to check if a move is a capture (when source and destination are 2 nodes apart)
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
                {renderPlayerPanel(false)}
              </div>
              <div className="bg-gray-800 rounded-lg p-2 flex gap-2">
                {gameStatus === "PLAYING" ? (
                  <>
                    <button
                      onClick={handleStopGame}
                      className="flex-1 px-3 py-2 bg-red-600 hover:bg-red-700 text-white text-sm font-medium rounded"
                    >
                      Stop Game
                    </button>
                    <button
                      onClick={undoMoves}
                      className={`flex-1 px-3 py-2 ${
                        canUndo && !isPaused
                          ? "bg-blue-600 hover:bg-blue-700"
                          : "bg-gray-600 cursor-not-allowed"
                      } text-white text-sm font-medium rounded`}
                      disabled={!canUndo || isPaused}
                    >
                      Undo Move
                    </button>
                  </>
                ) : (
                  <button
                    onClick={handleMainMenu}
                    className="w-full px-3 py-2 bg-gray-600 hover:bg-gray-700 text-white text-sm font-medium rounded"
                  >
                    Main Menu
                  </button>
                )}
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Modal for confirming stop game */}
      <Modal
        isOpen={isModalOpen}
        onClose={handleCancelStop}
        onConfirm={handleConfirmStop}
        title="End Game"
        message="Are you sure you want to exit to the main menu?"
      />
    </div>
  );
};

export default GameScreen;
