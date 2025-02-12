import React, { useState } from "react";
import { useNavigate } from "react-router-dom";
import { useGameStore } from "../stores/gameStore";
import useImage from "use-image";
import spriteGoat from "../assets/sprite_goat.png";
import spriteTiger from "../assets/sprite_tiger.png";

const TIME_PRESETS = [
  { name: "3+2", initial: 180, increment: 2 },
  { name: "5 min", initial: 300, increment: 0 },
  { name: "5+3", initial: 300, increment: 3 },
  { name: "10 min", initial: 600, increment: 0 },
  { name: "10|5", initial: 600, increment: 5 },
  { name: "15 min", initial: 900, increment: 0 },
  { name: "15|10", initial: 900, increment: 10 },
  { name: "Custom", initial: 600, increment: 0 },
];

const WelcomeScreen = () => {
  const navigate = useNavigate();
  const setGameSettings = useGameStore((state) => state.setGameSettings);
  const [spriteGoatImage] = useImage(spriteGoat);
  const [spriteTigerImage] = useImage(spriteTiger);

  const [settings, setSettings] = useState({
    players: {
      goat: "HUMAN",
      tiger: "HUMAN",
    },
    selectedPreset: TIME_PRESETS[4], // 10|5 default
    isCustom: false,
    customTime: {
      initial: 600,
      increment: 0,
      useIncrement: false,
    },
  });

  const handleStartGame = () => {
    const timeControl = settings.isCustom
      ? {
          initial: settings.customTime.initial,
          increment: settings.customTime.useIncrement
            ? settings.customTime.increment
            : 0,
        }
      : {
          initial: settings.selectedPreset.initial,
          increment: settings.selectedPreset.increment,
        };

    setGameSettings({
      players: settings.players,
      timeControl,
    });
    navigate("/game");
  };

  return (
    <div className="fixed inset-0 w-full h-full flex flex-col items-center justify-center bg-gradient-to-b from-gray-900 to-gray-800">
      <div className="max-w-4xl w-full mx-4 space-y-8 py-8 overflow-y-auto">
        {/* Game Title with Sprites */}
        <div className="text-center relative flex items-center justify-center mb-6">
          <div className="absolute left-1/2 -translate-x-[200px] md:-translate-x-[200px] -translate-x-[150px]">
            <img
              src={spriteGoat}
              alt="Goat"
              className="w-12 h-12 md:w-16 md:h-16"
            />
          </div>
          <div className="py-2">
            <h1 className="text-6xl md:text-6xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-yellow-400 to-yellow-600 block pb-2">
              Baghchal
            </h1>
          </div>
          <div className="absolute left-1/2 translate-x-[100px] md:translate-x-[140px]">
            <img
              src={spriteTiger}
              alt="Tiger"
              className="w-12 h-12 md:w-16 md:h-16"
            />
          </div>
        </div>

        {/* Game Modes */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {/* Player Selection */}
          <div className="bg-gray-800 p-6 rounded-xl border border-gray-700">
            <h2 className="text-xl font-bold text-white mb-4">Players</h2>
            <div className="space-y-4">
              {/* Goat Player Selection */}
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <img src={spriteGoat} alt="Goat" className="w-8 h-8" />
                  <span className="text-gray-300">Goat</span>
                </div>
                <div className="relative w-36">
                  <select
                    className="bg-gray-700 text-white rounded px-3 py-2 pr-10 appearance-none w-full"
                    value={settings.players.goat}
                    onChange={(e) =>
                      setSettings({
                        ...settings,
                        players: { ...settings.players, goat: e.target.value },
                      })
                    }
                  >
                    <option value="HUMAN">Human</option>
                    <option value="AI">Computer</option>
                  </select>
                  <span className="absolute right-3 top-1/2 transform -translate-y-1/2 text-gray-300 pointer-events-none">
                    ▼
                  </span>
                </div>
              </div>

              {/* Tiger Player Selection */}
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <img src={spriteTiger} alt="Tiger" className="w-8 h-8" />
                  <span className="text-gray-300">Tiger</span>
                </div>
                <div className="relative w-36">
                  <select
                    className="bg-gray-700 text-white rounded px-3 py-2 pr-10 appearance-none w-full"
                    value={settings.players.tiger}
                    onChange={(e) =>
                      setSettings({
                        ...settings,
                        players: { ...settings.players, tiger: e.target.value },
                      })
                    }
                  >
                    <option value="HUMAN">Human</option>
                    <option value="AI">Computer</option>
                  </select>
                  <span className="absolute right-3 top-1/2 transform -translate-y-1/2 text-gray-300 pointer-events-none">
                    ▼
                  </span>
                </div>
              </div>
            </div>
          </div>

          {/* Time Controls (Restored) */}
          <div className="bg-gray-800 p-6 rounded-xl border border-gray-700">
            <h2 className="text-xl font-bold text-white mb-4">Time Control</h2>
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
              {TIME_PRESETS.map((preset) => (
                <button
                  key={preset.name}
                  className={`p-2 rounded ${
                    settings.selectedPreset.name === preset.name &&
                    !settings.isCustom
                      ? "bg-blue-600 text-white"
                      : "bg-gray-700 text-gray-300 hover:bg-gray-600"
                  }`}
                  onClick={() => {
                    setSettings({
                      ...settings,
                      selectedPreset: preset,
                      isCustom: preset.name === "Custom",
                    });
                  }}
                >
                  {preset.name}
                </button>
              ))}
            </div>

            {settings.isCustom && (
              <div className="mt-4 space-y-4">
                <div>
                  <label className="text-gray-300 block mb-2">
                    Initial Time (minutes)
                  </label>
                  <input
                    type="number"
                    className="w-full bg-gray-700 text-white rounded px-3 py-2"
                    value={settings.customTime.initial / 60}
                    onChange={(e) =>
                      setSettings({
                        ...settings,
                        customTime: {
                          ...settings.customTime,
                          initial: Math.max(1, parseInt(e.target.value)) * 60,
                        },
                      })
                    }
                    min="1"
                  />
                </div>

                <div className="flex items-center gap-2">
                  <input
                    type="checkbox"
                    id="useIncrement"
                    className="rounded bg-gray-700 border-gray-600 text-blue-600 focus:ring-blue-500"
                    checked={settings.customTime.useIncrement}
                    onChange={(e) =>
                      setSettings({
                        ...settings,
                        customTime: {
                          ...settings.customTime,
                          useIncrement: e.target.checked,
                          increment: e.target.checked ? 5 : 0,
                        },
                      })
                    }
                  />
                  <label htmlFor="useIncrement" className="text-gray-300">
                    Add increment
                  </label>
                </div>

                {settings.customTime.useIncrement && (
                  <div>
                    <label className="text-gray-300 block mb-2">
                      Increment (seconds)
                    </label>
                    <input
                      type="number"
                      className="w-full bg-gray-700 text-white rounded px-3 py-2"
                      value={settings.customTime.increment}
                      onChange={(e) =>
                        setSettings({
                          ...settings,
                          customTime: {
                            ...settings.customTime,
                            increment: Math.max(0, parseInt(e.target.value)),
                          },
                        })
                      }
                      min="0"
                    />
                  </div>
                )}
              </div>
            )}
          </div>
        </div>

        {/* Start Button */}
        <div className="flex justify-center">
          <button
            className="w-full max-w-md px-8 py-4 bg-gradient-to-r from-yellow-500 to-yellow-600 
                       text-white text-xl font-bold rounded-lg shadow-lg
                       hover:from-yellow-600 hover:to-yellow-700 
                       transform transition-all duration-200 hover:scale-105"
            onClick={handleStartGame}
          >
            Start Game
          </button>
        </div>
      </div>
    </div>
  );
};

export default WelcomeScreen;
