import React, { useState } from "react";
import { useNavigate } from "react-router-dom";
import { useGameStore } from "../stores/gameStore";
import useImage from "use-image";
import spriteGoat from "../assets/sprite_goat.png";
import spriteTiger from "../assets/sprite_tiger.png";
import CustomSelect from "../components/CustomSelect";

const TIME_PRESETS = [
  { name: "3m | 2s", initial: 180, increment: 2 },
  { name: "5 min", initial: 300, increment: 0 },
  { name: "5m | 3s", initial: 300, increment: 3 },
  { name: "10 min", initial: 600, increment: 0 },
  { name: "10m | 5s", initial: 600, increment: 5 },
  { name: "15 min", initial: 900, increment: 0 },
  { name: "15m | 10s", initial: 900, increment: 10 },
  { name: "Custom", initial: 600, increment: 0 },
];

// Default agent settings
const DEFAULT_AGENT_SETTINGS = {
  minimax: {
    max_depth: 6,
    randomize_equal_moves: true,
    useTunedParams: false,
  },
  mcts: {
    iterations: 20000,
    exploration_weight: 1.414,
    rollout_policy: "lightweight",
    guided_strictness: 0.7,
    max_rollout_depth: 6,
    max_time_seconds: 50,
  },
  "q-learning": {
    tables_path: "backend/simulation_results/q_tables",
  },
};

// Options for rollout policies
const ROLLOUT_POLICIES = [
  { value: "random", label: "Random" },
  { value: "guided", label: "Guided" },
  { value: "lightweight", label: "Lightweight" },
];

const WelcomeScreen = () => {
  const navigate = useNavigate();
  const setGameSettings = useGameStore((state) => state.setGameSettings);

  const [settings, setSettings] = useState({
    players: {
      goat: { type: "HUMAN", model: null, settings: null },
      tiger: { type: "HUMAN", model: null, settings: null },
    },
    selectedPreset: TIME_PRESETS[4], // 10|5 default
    isCustom: false,
    customTime: {
      initial: 600,
      increment: 0,
      useIncrement: false,
    },
    showAdvancedSettings: false,
  });

  // Helper function to update agent settings
  const updateAgentSettings = (player, key, value) => {
    setSettings((prevSettings) => {
      // Clone the current settings for the player
      const playerSettings = {
        ...(prevSettings.players[player].settings ||
          (prevSettings.players[player].model === "minimax"
            ? { ...DEFAULT_AGENT_SETTINGS.minimax }
            : { ...DEFAULT_AGENT_SETTINGS.mcts })),
      };

      // Update the specific setting
      playerSettings[key] = value;

      return {
        ...prevSettings,
        players: {
          ...prevSettings.players,
          [player]: {
            ...prevSettings.players[player],
            settings: playerSettings,
          },
        },
      };
    });
  };

  // Function to initialize settings when model changes
  const initializeSettings = (player, model) => {
    if (model) {
      const defaultSettings =
        model === "minimax"
          ? { ...DEFAULT_AGENT_SETTINGS.minimax }
          : model === "mcts"
          ? { ...DEFAULT_AGENT_SETTINGS.mcts }
          : model === "q-learning"
          ? { ...DEFAULT_AGENT_SETTINGS["q-learning"] }
          : null;

      setSettings((prevSettings) => ({
        ...prevSettings,
        players: {
          ...prevSettings.players,
          [player]: {
            ...prevSettings.players[player],
            model,
            settings: defaultSettings,
          },
        },
      }));
    } else {
      // When switching back to human player
      setSettings((prevSettings) => ({
        ...prevSettings,
        players: {
          ...prevSettings.players,
          [player]: {
            ...prevSettings.players[player],
            model: null,
            settings: null,
          },
        },
      }));
    }
  };

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

    // Determine perspective based on human player
    let perspective = "GOAT"; // default
    if (
      settings.players.goat.type === "AI" &&
      settings.players.tiger.type === "HUMAN"
    ) {
      perspective = "TIGER";
    } else if (
      settings.players.goat.type === "HUMAN" &&
      settings.players.tiger.type === "HUMAN"
    ) {
      perspective = "GOAT"; // In case both are human, keep GOAT perspective
    }

    setGameSettings({
      players: settings.players,
      timeControl,
      perspective,
    });
    navigate("/game");
  };

  return (
    <div className="fixed inset-0 w-full h-full flex flex-col items-center justify-center bg-gradient-to-b from-gray-900 to-gray-800">
      <div className="max-w-6xl w-full mx-auto px-4 space-y-8 py-8 overflow-y-auto">
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
        <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
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
                <div className="flex gap-2">
                  <CustomSelect
                    value={settings.players.goat.type}
                    onChange={(e) =>
                      setSettings({
                        ...settings,
                        players: {
                          ...settings.players,
                          goat: {
                            ...settings.players.goat,
                            type: e.target.value,
                            model: e.target.value === "AI" ? "mcts" : null,
                          },
                        },
                      })
                    }
                    options={[
                      { value: "HUMAN", label: "Human" },
                      { value: "AI", label: "Computer" },
                    ]}
                  />
                  {settings.players.goat.type === "AI" && (
                    <CustomSelect
                      value={settings.players.goat.model}
                      onChange={(e) =>
                        initializeSettings("goat", e.target.value)
                      }
                      options={[
                        { value: "minimax", label: "Minimax" },
                        { value: "mcts", label: "MCTS" },
                        { value: "random", label: "Random" },
                        { value: "q-learning", label: "Q-Learning" },
                      ]}
                    />
                  )}
                </div>
              </div>
              {/* Goat Agent Settings */}
              {settings.players.goat.type === "AI" &&
                settings.showAdvancedSettings && (
                  <div className="mt-3 mb-4 bg-gray-750 p-3 rounded-lg">
                    <h3 className="text-sm font-medium text-gray-300 mb-2">
                      Goat AI Settings
                    </h3>
                    {settings.players.goat.model === "minimax" ? (
                      <div className="grid grid-cols-2 gap-4">
                        <div>
                          <label className="text-gray-300 block text-sm mb-1">
                            Search Depth (1-9)
                          </label>
                          <input
                            type="number"
                            min="1"
                            max="9"
                            value={
                              settings.players.goat.settings?.max_depth ||
                              DEFAULT_AGENT_SETTINGS.minimax.max_depth
                            }
                            onChange={(e) =>
                              updateAgentSettings(
                                "goat",
                                "max_depth",
                                Math.min(
                                  9,
                                  Math.max(1, parseInt(e.target.value) || 1)
                                )
                              )
                            }
                            className="w-full bg-gray-700 text-white rounded px-2 py-1 text-sm"
                          />
                        </div>
                        <div>
                          <label className="text-gray-300 block text-sm mb-1">
                            Randomize Equal Moves
                          </label>
                          <div className="h-[26px] flex items-center">
                            <input
                              type="checkbox"
                              id="randomize-goat"
                              checked={
                                settings.players.goat.settings
                                  ?.randomize_equal_moves ||
                                DEFAULT_AGENT_SETTINGS.minimax
                                  .randomize_equal_moves
                              }
                              onChange={(e) =>
                                updateAgentSettings(
                                  "goat",
                                  "randomize_equal_moves",
                                  e.target.checked
                                )
                              }
                              className="bg-gray-700 w-4 h-4"
                            />
                            <label
                              htmlFor="randomize-goat"
                              className="text-gray-300 text-sm ml-2"
                            >
                              Enabled
                            </label>
                          </div>
                        </div>
                        <div>
                          <label className="text-gray-300 block text-sm mb-1">
                            Use Tuned Parameters
                          </label>
                          <div className="h-[26px] flex items-center">
                            <input
                              type="checkbox"
                              id="tuned-params-goat"
                              checked={
                                settings.players.goat.settings
                                  ?.useTunedParams ||
                                DEFAULT_AGENT_SETTINGS.minimax.useTunedParams
                              }
                              onChange={(e) =>
                                updateAgentSettings(
                                  "goat",
                                  "useTunedParams",
                                  e.target.checked
                                )
                              }
                              className="bg-gray-700 w-4 h-4"
                            />
                            <label
                              htmlFor="tuned-params-goat"
                              className="text-gray-300 text-sm ml-2"
                            >
                              Enabled
                            </label>
                          </div>
                        </div>
                      </div>
                    ) : settings.players.goat.model === "mcts" ? (
                      <div className="grid grid-cols-2 gap-4">
                        <div>
                          <label className="text-gray-300 block text-sm mb-1">
                            Iterations (100-100000)
                          </label>
                          <input
                            type="number"
                            min="100"
                            max="100000"
                            step="100"
                            value={
                              settings.players.goat.settings?.iterations ||
                              DEFAULT_AGENT_SETTINGS.mcts.iterations
                            }
                            onChange={(e) =>
                              updateAgentSettings(
                                "goat",
                                "iterations",
                                Math.min(
                                  100000,
                                  Math.max(100, parseInt(e.target.value) || 100)
                                )
                              )
                            }
                            className="w-full bg-gray-700 text-white rounded px-2 py-1 text-sm"
                          />
                        </div>
                        <div>
                          <label className="text-gray-300 block text-sm mb-1">
                            Max Time (seconds)
                          </label>
                          <input
                            type="number"
                            min="1"
                            max="300"
                            value={
                              settings.players.goat.settings
                                ?.max_time_seconds ||
                              DEFAULT_AGENT_SETTINGS.mcts.max_time_seconds
                            }
                            onChange={(e) =>
                              updateAgentSettings(
                                "goat",
                                "max_time_seconds",
                                Math.min(
                                  300,
                                  Math.max(1, parseInt(e.target.value) || 1)
                                )
                              )
                            }
                            className="w-full bg-gray-700 text-white rounded px-2 py-1 text-sm"
                          />
                        </div>
                        <div>
                          <label className="text-gray-300 block text-sm mb-1">
                            Rollout Policy
                          </label>
                          <select
                            value={
                              settings.players.goat.settings?.rollout_policy ||
                              DEFAULT_AGENT_SETTINGS.mcts.rollout_policy
                            }
                            onChange={(e) =>
                              updateAgentSettings(
                                "goat",
                                "rollout_policy",
                                e.target.value
                              )
                            }
                            className="w-full bg-gray-700 text-white rounded px-2 py-1 text-sm"
                          >
                            {ROLLOUT_POLICIES.map((policy) => (
                              <option key={policy.value} value={policy.value}>
                                {policy.label}
                              </option>
                            ))}
                          </select>
                        </div>
                        <div>
                          <label className="text-gray-300 block text-sm mb-1">
                            Max Rollout Depth
                          </label>
                          <input
                            type="number"
                            min="1"
                            max="20"
                            value={
                              settings.players.goat.settings
                                ?.max_rollout_depth ||
                              DEFAULT_AGENT_SETTINGS.mcts.max_rollout_depth
                            }
                            onChange={(e) =>
                              updateAgentSettings(
                                "goat",
                                "max_rollout_depth",
                                Math.min(
                                  20,
                                  Math.max(1, parseInt(e.target.value) || 1)
                                )
                              )
                            }
                            className="w-full bg-gray-700 text-white rounded px-2 py-1 text-sm"
                          />
                        </div>
                      </div>
                    ) : settings.players.goat.model === "q-learning" ? (
                      <div className="grid grid-cols-2 gap-4">
                        <div className="col-span-2">
                          <label className="text-gray-300 block text-sm mb-1">
                            Q-Tables Path
                          </label>
                          <input
                            type="text"
                            value={
                              settings.players.goat.settings?.tables_path ||
                              DEFAULT_AGENT_SETTINGS["q-learning"].tables_path
                            }
                            onChange={(e) =>
                              updateAgentSettings(
                                "goat",
                                "tables_path",
                                e.target.value
                              )
                            }
                            className="w-full bg-gray-700 text-white rounded px-2 py-1 text-sm"
                          />
                          <p className="text-xs text-gray-400 mt-1">
                            Path to directory containing trained Q-tables
                          </p>
                        </div>
                      </div>
                    ) : null}
                  </div>
                )}

              {/* Divider between sections */}
              {settings.players.goat.type === "AI" &&
                settings.players.tiger.type === "AI" &&
                settings.showAdvancedSettings && (
                  <div className="my-3 border-t border-gray-700"></div>
                )}

              {/* Tiger Player Selection */}
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <img src={spriteTiger} alt="Tiger" className="w-8 h-8" />
                  <span className="text-gray-300">Tiger</span>
                </div>
                <div className="flex gap-2">
                  <CustomSelect
                    value={settings.players.tiger.type}
                    onChange={(e) =>
                      setSettings({
                        ...settings,
                        players: {
                          ...settings.players,
                          tiger: {
                            ...settings.players.tiger,
                            type: e.target.value,
                            model: e.target.value === "AI" ? "mcts" : null,
                          },
                        },
                      })
                    }
                    options={[
                      { value: "HUMAN", label: "Human" },
                      { value: "AI", label: "Computer" },
                    ]}
                  />
                  {settings.players.tiger.type === "AI" && (
                    <CustomSelect
                      value={settings.players.tiger.model}
                      onChange={(e) =>
                        initializeSettings("tiger", e.target.value)
                      }
                      options={[
                        { value: "minimax", label: "Minimax" },
                        { value: "mcts", label: "MCTS" },
                        { value: "random", label: "Random" },
                        { value: "q-learning", label: "Q-Learning" },
                      ]}
                    />
                  )}
                </div>
              </div>
              {/* Tiger Agent Settings */}
              {settings.players.tiger.type === "AI" &&
                settings.showAdvancedSettings && (
                  <div className="mt-3 bg-gray-750 p-3 rounded-lg">
                    <h3 className="text-sm font-medium text-gray-300 mb-2">
                      Tiger AI Settings
                    </h3>
                    {settings.players.tiger.model === "minimax" ? (
                      <div className="grid grid-cols-2 gap-4">
                        <div>
                          <label className="text-gray-300 block text-sm mb-1">
                            Search Depth (1-9)
                          </label>
                          <input
                            type="number"
                            min="1"
                            max="9"
                            value={
                              settings.players.tiger.settings?.max_depth ||
                              DEFAULT_AGENT_SETTINGS.minimax.max_depth
                            }
                            onChange={(e) =>
                              updateAgentSettings(
                                "tiger",
                                "max_depth",
                                Math.min(
                                  9,
                                  Math.max(1, parseInt(e.target.value) || 1)
                                )
                              )
                            }
                            className="w-full bg-gray-700 text-white rounded px-2 py-1 text-sm"
                          />
                        </div>
                        <div>
                          <label className="text-gray-300 block text-sm mb-1">
                            Randomize Equal Moves
                          </label>
                          <div className="h-[26px] flex items-center">
                            <input
                              type="checkbox"
                              id="randomize-tiger"
                              checked={
                                settings.players.tiger.settings
                                  ?.randomize_equal_moves ||
                                DEFAULT_AGENT_SETTINGS.minimax
                                  .randomize_equal_moves
                              }
                              onChange={(e) =>
                                updateAgentSettings(
                                  "tiger",
                                  "randomize_equal_moves",
                                  e.target.checked
                                )
                              }
                              className="bg-gray-700 w-4 h-4"
                            />
                            <label
                              htmlFor="randomize-tiger"
                              className="text-gray-300 text-sm ml-2"
                            >
                              Enabled
                            </label>
                          </div>
                        </div>
                        <div>
                          <label className="text-gray-300 block text-sm mb-1">
                            Use Tuned Parameters
                          </label>
                          <div className="h-[26px] flex items-center">
                            <input
                              type="checkbox"
                              id="tuned-params-tiger"
                              checked={
                                settings.players.tiger.settings
                                  ?.useTunedParams ||
                                DEFAULT_AGENT_SETTINGS.minimax.useTunedParams
                              }
                              onChange={(e) =>
                                updateAgentSettings(
                                  "tiger",
                                  "useTunedParams",
                                  e.target.checked
                                )
                              }
                              className="bg-gray-700 w-4 h-4"
                            />
                            <label
                              htmlFor="tuned-params-tiger"
                              className="text-gray-300 text-sm ml-2"
                            >
                              Enabled
                            </label>
                          </div>
                        </div>
                      </div>
                    ) : settings.players.tiger.model === "mcts" ? (
                      <div className="grid grid-cols-2 gap-4">
                        <div>
                          <label className="text-gray-300 block text-sm mb-1">
                            Iterations (100-100000)
                          </label>
                          <input
                            type="number"
                            min="100"
                            max="100000"
                            step="100"
                            value={
                              settings.players.tiger.settings?.iterations ||
                              DEFAULT_AGENT_SETTINGS.mcts.iterations
                            }
                            onChange={(e) =>
                              updateAgentSettings(
                                "tiger",
                                "iterations",
                                Math.min(
                                  100000,
                                  Math.max(100, parseInt(e.target.value) || 100)
                                )
                              )
                            }
                            className="w-full bg-gray-700 text-white rounded px-2 py-1 text-sm"
                          />
                        </div>
                        <div>
                          <label className="text-gray-300 block text-sm mb-1">
                            Max Time (seconds)
                          </label>
                          <input
                            type="number"
                            min="1"
                            max="300"
                            value={
                              settings.players.tiger.settings
                                ?.max_time_seconds ||
                              DEFAULT_AGENT_SETTINGS.mcts.max_time_seconds
                            }
                            onChange={(e) =>
                              updateAgentSettings(
                                "tiger",
                                "max_time_seconds",
                                Math.min(
                                  300,
                                  Math.max(1, parseInt(e.target.value) || 1)
                                )
                              )
                            }
                            className="w-full bg-gray-700 text-white rounded px-2 py-1 text-sm"
                          />
                        </div>
                        <div>
                          <label className="text-gray-300 block text-sm mb-1">
                            Rollout Policy
                          </label>
                          <select
                            value={
                              settings.players.tiger.settings?.rollout_policy ||
                              DEFAULT_AGENT_SETTINGS.mcts.rollout_policy
                            }
                            onChange={(e) =>
                              updateAgentSettings(
                                "tiger",
                                "rollout_policy",
                                e.target.value
                              )
                            }
                            className="w-full bg-gray-700 text-white rounded px-2 py-1 text-sm"
                          >
                            {ROLLOUT_POLICIES.map((policy) => (
                              <option key={policy.value} value={policy.value}>
                                {policy.label}
                              </option>
                            ))}
                          </select>
                        </div>
                        <div>
                          <label className="text-gray-300 block text-sm mb-1">
                            Max Rollout Depth
                          </label>
                          <input
                            type="number"
                            min="1"
                            max="20"
                            value={
                              settings.players.tiger.settings
                                ?.max_rollout_depth ||
                              DEFAULT_AGENT_SETTINGS.mcts.max_rollout_depth
                            }
                            onChange={(e) =>
                              updateAgentSettings(
                                "tiger",
                                "max_rollout_depth",
                                Math.min(
                                  20,
                                  Math.max(1, parseInt(e.target.value) || 1)
                                )
                              )
                            }
                            className="w-full bg-gray-700 text-white rounded px-2 py-1 text-sm"
                          />
                        </div>
                      </div>
                    ) : settings.players.tiger.model === "q-learning" ? (
                      <div className="grid grid-cols-2 gap-4">
                        <div className="col-span-2">
                          <label className="text-gray-300 block text-sm mb-1">
                            Q-Tables Path
                          </label>
                          <input
                            type="text"
                            value={
                              settings.players.tiger.settings?.tables_path ||
                              DEFAULT_AGENT_SETTINGS["q-learning"].tables_path
                            }
                            onChange={(e) =>
                              updateAgentSettings(
                                "tiger",
                                "tables_path",
                                e.target.value
                              )
                            }
                            className="w-full bg-gray-700 text-white rounded px-2 py-1 text-sm"
                          />
                          <p className="text-xs text-gray-400 mt-1">
                            Path to directory containing trained Q-tables
                          </p>
                        </div>
                      </div>
                    ) : null}
                  </div>
                )}

              {/* Toggle Advanced Settings */}
              {(settings.players.goat.type === "AI" ||
                settings.players.tiger.type === "AI") && (
                <div className="mt-3 flex items-center">
                  <button
                    onClick={() =>
                      setSettings({
                        ...settings,
                        showAdvancedSettings: !settings.showAdvancedSettings,
                      })
                    }
                    className="text-sm text-blue-400 hover:text-blue-300 flex items-center"
                  >
                    {settings.showAdvancedSettings ? "Hide" : "Show"} Advanced
                    Settings
                    <svg
                      xmlns="http://www.w3.org/2000/svg"
                      className={`h-4 w-4 ml-1 transition-transform ${
                        settings.showAdvancedSettings ? "rotate-180" : ""
                      }`}
                      fill="none"
                      viewBox="0 0 24 24"
                      stroke="currentColor"
                    >
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth={2}
                        d="M19 9l-7 7-7-7"
                      />
                    </svg>
                  </button>
                </div>
              )}
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
                          initial: parseInt(e.target.value) * 60,
                        },
                      })
                    }
                  />
                </div>

                <div className="flex items-center gap-2">
                  <input
                    type="checkbox"
                    id="use-increment"
                    className="bg-gray-700"
                    checked={settings.customTime.useIncrement}
                    onChange={(e) =>
                      setSettings({
                        ...settings,
                        customTime: {
                          ...settings.customTime,
                          useIncrement: e.target.checked,
                        },
                      })
                    }
                  />
                  <label htmlFor="use-increment" className="text-gray-300">
                    Use Increment
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
                            increment: parseInt(e.target.value),
                          },
                        })
                      }
                    />
                  </div>
                )}
              </div>
            )}
          </div>
        </div>

        {/* Action Buttons */}
        <div className="flex justify-center gap-4 mt-6">
          <button
            onClick={handleStartGame}
            className="px-6 py-3 rounded-lg bg-yellow-500 hover:bg-yellow-600 text-gray-900 font-bold transition-colors"
          >
            Start Game
          </button>
        </div>
      </div>
    </div>
  );
};

export default WelcomeScreen;
