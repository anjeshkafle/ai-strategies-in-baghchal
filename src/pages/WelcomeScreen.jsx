import React, { useState } from "react";
import { useNavigate } from "react-router-dom";
import { useGameStore } from "../stores/gameStore";

const WelcomeScreen = () => {
  const navigate = useNavigate();
  const setGameSettings = useGameStore((state) => state.setGameSettings);

  const [settings, setSettings] = useState({
    players: {
      goat: "HUMAN",
      tiger: "HUMAN",
    },
    timeControl: {
      initial: 600,
      increment: 5,
    },
    useIncrement: true,
  });

  const handleStartGame = () => {
    const finalSettings = {
      players: settings.players,
      timeControl: {
        initial: settings.timeControl.initial,
        increment: settings.useIncrement ? settings.timeControl.increment : 0,
      },
    };
    setGameSettings(finalSettings);
    navigate("/game");
  };

  return (
    <div className="fixed inset-0 w-full h-full flex flex-col items-center justify-center bg-gray-900">
      <div className="bg-gray-800 p-8 rounded-lg shadow-xl max-w-md w-full mx-4">
        <h1 className="text-4xl font-bold text-white mb-8 text-center">
          Baghchal
        </h1>

        {/* Player Settings */}
        <div className="space-y-6 mb-8">
          <div>
            <label className="text-gray-300 block mb-2">Goat Player</label>
            <select
              className="w-full bg-gray-700 text-white rounded px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
              value={settings.players.goat}
              onChange={(e) =>
                setSettings({
                  ...settings,
                  players: { ...settings.players, goat: e.target.value },
                })
              }
            >
              <option value="HUMAN">Human</option>
              <option value="AI">AI</option>
            </select>
          </div>

          <div>
            <label className="text-gray-300 block mb-2">Tiger Player</label>
            <select
              className="w-full bg-gray-700 text-white rounded px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
              value={settings.players.tiger}
              onChange={(e) =>
                setSettings({
                  ...settings,
                  players: { ...settings.players, tiger: e.target.value },
                })
              }
            >
              <option value="HUMAN">Human</option>
              <option value="AI">AI</option>
            </select>
          </div>
        </div>

        {/* Time Controls */}
        <div className="space-y-4 mb-8">
          <div>
            <label className="text-gray-300 block mb-2">
              Initial Time (minutes)
            </label>
            <input
              type="number"
              className="w-full bg-gray-700 text-white rounded px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
              value={settings.timeControl.initial / 60}
              onChange={(e) =>
                setSettings({
                  ...settings,
                  timeControl: {
                    ...settings.timeControl,
                    initial: Math.max(1, parseInt(e.target.value)) * 60,
                  },
                })
              }
              min="1"
            />
          </div>

          <div className="flex items-center">
            <input
              type="checkbox"
              id="useIncrement"
              className="mr-2"
              checked={settings.useIncrement}
              onChange={(e) =>
                setSettings({
                  ...settings,
                  useIncrement: e.target.checked,
                })
              }
            />
            <label htmlFor="useIncrement" className="text-gray-300">
              Use Increment
            </label>
          </div>

          {settings.useIncrement && (
            <div>
              <label className="text-gray-300 block mb-2">
                Increment (seconds)
              </label>
              <input
                type="number"
                className="w-full bg-gray-700 text-white rounded px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
                value={settings.timeControl.increment}
                onChange={(e) =>
                  setSettings({
                    ...settings,
                    timeControl: {
                      ...settings.timeControl,
                      increment: Math.max(0, parseInt(e.target.value)),
                    },
                  })
                }
                min="0"
              />
            </div>
          )}
        </div>

        <button
          className="w-full px-6 py-3 bg-blue-600 text-white text-lg font-semibold rounded-lg 
                     hover:bg-blue-700 transform transition-all duration-200"
          onClick={handleStartGame}
        >
          Start Game
        </button>
      </div>
    </div>
  );
};

export default WelcomeScreen;
