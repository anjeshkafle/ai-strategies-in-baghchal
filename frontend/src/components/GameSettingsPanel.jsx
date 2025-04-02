import React, { useState, useEffect } from "react";
import { useGameStore } from "../stores/gameStore";
import CustomSelect from "./CustomSelect";
import useImage from "use-image";
import spriteGoat from "../assets/sprite_goat.png";
import spriteTiger from "../assets/sprite_tiger.png";

// Default agent settings (copied from WelcomeScreen)
const DEFAULT_AGENT_SETTINGS = {
  minimax: {
    max_depth: 6,
    randomize_equal_moves: true,
  },
  mcts: {
    iterations: 20000,
    exploration_weight: 1.414,
    rollout_policy: "lightweight",
    guided_strictness: 0.7,
    max_rollout_depth: 6,
    max_time_seconds: 50,
  },
};

// Options for rollout policies
const ROLLOUT_POLICIES = [
  { value: "random", label: "Random" },
  { value: "guided", label: "Guided" },
  { value: "lightweight", label: "Lightweight" },
];

// Time presets
const TIME_PRESETS = [
  { name: "Unchanged", initial: null, increment: null },
  { name: "3m | 2s", initial: 180, increment: 2 },
  { name: "5 min", initial: 300, increment: 0 },
  { name: "5m | 3s", initial: 300, increment: 3 },
  { name: "10 min", initial: 600, increment: 0 },
  { name: "10m | 5s", initial: 600, increment: 5 },
  { name: "15 min", initial: 900, increment: 0 },
  { name: "15m | 10s", initial: 900, increment: 10 },
  { name: "Custom", initial: 600, increment: 0 },
];

const GameSettingsPanel = ({ isPaused, onTogglePause, onApplySettings }) => {
  const { players, timeControl, gameStatus } = useGameStore();

  // Track if settings have changed
  const [hasChanges, setHasChanges] = useState(false);

  // Local state for editing settings
  const [settings, setSettings] = useState({
    players: {
      goat: { type: "HUMAN", model: null, settings: null },
      tiger: { type: "HUMAN", model: null, settings: null },
    },
    selectedPreset: TIME_PRESETS[0], // Default to "Unchanged"
    isCustom: false,
    customTime: {
      initial: timeControl.initial,
      increment: timeControl.increment,
      useIncrement: timeControl.increment > 0,
    },
    showAdvancedSettings: false,
  });

  // Store original settings to detect changes
  const [originalSettings, setOriginalSettings] = useState(null);

  // Sync settings with game store when component mounts or players/timeControl changes
  useEffect(() => {
    const updatedSettings = {
      players: JSON.parse(JSON.stringify(players)), // Deep copy
      selectedPreset: TIME_PRESETS[0], // Default to "Unchanged"
      isCustom: false,
      customTime: {
        initial: timeControl.initial,
        increment: timeControl.increment,
        useIncrement: timeControl.increment > 0,
      },
      showAdvancedSettings: false,
    };

    setSettings(updatedSettings);

    // If newly paused, store original settings
    if (isPaused && !originalSettings) {
      setOriginalSettings(JSON.parse(JSON.stringify(updatedSettings)));
      setHasChanges(false);
    } else if (!isPaused) {
      setOriginalSettings(null);
      setHasChanges(false);
    }
  }, [players, timeControl, isPaused]);

  // Check for changes whenever settings update
  useEffect(() => {
    if (!originalSettings) return;

    const playersChanged =
      JSON.stringify(settings.players) !==
      JSON.stringify(originalSettings.players);

    const timeChanged =
      (settings.isCustom &&
        (settings.customTime.initial !== originalSettings.customTime.initial ||
          settings.customTime.increment !==
            originalSettings.customTime.increment)) ||
      (settings.selectedPreset.name !== "Unchanged" && !settings.isCustom);

    setHasChanges(playersChanged || timeChanged);
  }, [settings, originalSettings]);

  // Update settings helper functions
  const updateAgentSettings = (player, key, value) => {
    if (!isPaused) return;

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
    if (!isPaused) return;

    if (model) {
      const defaultSettings =
        model === "minimax"
          ? { ...DEFAULT_AGENT_SETTINGS.minimax }
          : model === "mcts"
          ? { ...DEFAULT_AGENT_SETTINGS.mcts }
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

  // Handler for applying settings or just resuming
  const handleApplyOrResume = () => {
    if (hasChanges) {
      // Apply settings and resume
      let updatedTimeControl;

      if (settings.selectedPreset.name === "Unchanged") {
        // Keep the original time controls
        updatedTimeControl = { ...timeControl };
        console.log("Using unchanged time control:", updatedTimeControl);
      } else if (settings.isCustom) {
        updatedTimeControl = {
          initial: settings.customTime.initial,
          increment: settings.customTime.useIncrement
            ? settings.customTime.increment
            : 0,
        };
        console.log("Using custom time control:", updatedTimeControl);
      } else {
        updatedTimeControl = {
          initial: settings.selectedPreset.initial,
          increment: settings.selectedPreset.increment,
        };
        console.log(
          "Using preset time control:",
          updatedTimeControl,
          "from preset:",
          settings.selectedPreset
        );
      }

      // Log what we're applying
      console.log("Applying settings:", {
        players: settings.players,
        timeControl: updatedTimeControl,
      });

      onApplySettings({
        players: settings.players,
        timeControl: updatedTimeControl,
      });

      // Reset changes state after applying
      setHasChanges(false);

      // Update original settings to reflect the new state
      setOriginalSettings(JSON.parse(JSON.stringify(settings)));
    } else {
      // Just resume without applying changes
      onTogglePause();
    }
  };

  return (
    <div className="bg-gray-800 rounded-lg w-full h-full flex flex-col overflow-hidden">
      {/* Header with Pause/Resume button */}
      <div className="p-4 border-b border-gray-700 flex items-center justify-between">
        <h2 className="text-lg font-bold text-white">Game Settings</h2>
        {!isPaused && (
          <button
            onClick={onTogglePause}
            className="px-4 py-2 bg-yellow-600 hover:bg-yellow-700 text-white rounded text-sm font-medium"
            disabled={gameStatus !== "PLAYING"}
          >
            Pause Game
          </button>
        )}
      </div>

      {/* Settings content - scrollable */}
      <div className="flex-grow overflow-y-auto p-4 space-y-6">
        {!isPaused && (
          <div className="p-4 bg-gray-750 rounded-lg text-center text-gray-400">
            Settings can be changed when the game is paused.
          </div>
        )}

        {/* Player Selection */}
        <div className="space-y-4">
          <h3 className="text-md font-bold text-white border-b border-gray-700 pb-2">
            Players
          </h3>

          {/* Goat Player Selection */}
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <img src={spriteGoat} alt="Goat" className="w-8 h-8" />
              <span className="text-gray-300">Goat</span>
            </div>
            <div className="flex gap-2">
              <CustomSelect
                value={settings.players.goat.type}
                onChange={(e) => {
                  if (!isPaused) return;
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
                  });
                }}
                options={[
                  { value: "HUMAN", label: "Human" },
                  { value: "AI", label: "Computer" },
                ]}
                disabled={!isPaused}
              />
              {settings.players.goat.type === "AI" && (
                <CustomSelect
                  value={settings.players.goat.model}
                  onChange={(e) => initializeSettings("goat", e.target.value)}
                  options={[
                    { value: "minimax", label: "Minimax Base" },
                    { value: "mcts", label: "MCTS Agent" },
                    { value: "random", label: "Random AI" },
                  ]}
                  disabled={!isPaused}
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
                        className={`w-full ${
                          isPaused
                            ? "bg-gray-700 text-white"
                            : "bg-gray-800 text-gray-500 cursor-not-allowed"
                        } rounded px-2 py-1 text-sm`}
                        disabled={!isPaused}
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
                            DEFAULT_AGENT_SETTINGS.minimax.randomize_equal_moves
                          }
                          onChange={(e) =>
                            updateAgentSettings(
                              "goat",
                              "randomize_equal_moves",
                              e.target.checked
                            )
                          }
                          className={`w-4 h-4 ${
                            isPaused
                              ? "bg-gray-700"
                              : "bg-gray-800 cursor-not-allowed"
                          }`}
                          disabled={!isPaused}
                        />
                        <label
                          htmlFor="randomize-goat"
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
                        className={`w-full ${
                          isPaused
                            ? "bg-gray-700 text-white"
                            : "bg-gray-800 text-gray-500 cursor-not-allowed"
                        } rounded px-2 py-1 text-sm`}
                        disabled={!isPaused}
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
                          settings.players.goat.settings?.max_time_seconds ||
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
                        className={`w-full ${
                          isPaused
                            ? "bg-gray-700 text-white"
                            : "bg-gray-800 text-gray-500 cursor-not-allowed"
                        } rounded px-2 py-1 text-sm`}
                        disabled={!isPaused}
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
                        className={`w-full ${
                          isPaused
                            ? "bg-gray-700 text-white"
                            : "bg-gray-800 text-gray-500 cursor-not-allowed"
                        } rounded px-2 py-1 text-sm`}
                        disabled={!isPaused}
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
                          settings.players.goat.settings?.max_rollout_depth ||
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
                        className={`w-full ${
                          isPaused
                            ? "bg-gray-700 text-white"
                            : "bg-gray-800 text-gray-500 cursor-not-allowed"
                        } rounded px-2 py-1 text-sm`}
                        disabled={!isPaused}
                      />
                    </div>
                  </div>
                ) : null}
              </div>
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
                onChange={(e) => {
                  if (!isPaused) return;
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
                  });
                }}
                options={[
                  { value: "HUMAN", label: "Human" },
                  { value: "AI", label: "Computer" },
                ]}
                disabled={!isPaused}
              />
              {settings.players.tiger.type === "AI" && (
                <CustomSelect
                  value={settings.players.tiger.model}
                  onChange={(e) => initializeSettings("tiger", e.target.value)}
                  options={[
                    { value: "minimax", label: "Minimax Base" },
                    { value: "mcts", label: "MCTS Agent" },
                    { value: "random", label: "Random AI" },
                  ]}
                  disabled={!isPaused}
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
                        className={`w-full ${
                          isPaused
                            ? "bg-gray-700 text-white"
                            : "bg-gray-800 text-gray-500 cursor-not-allowed"
                        } rounded px-2 py-1 text-sm`}
                        disabled={!isPaused}
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
                            DEFAULT_AGENT_SETTINGS.minimax.randomize_equal_moves
                          }
                          onChange={(e) =>
                            updateAgentSettings(
                              "tiger",
                              "randomize_equal_moves",
                              e.target.checked
                            )
                          }
                          className={`w-4 h-4 ${
                            isPaused
                              ? "bg-gray-700"
                              : "bg-gray-800 cursor-not-allowed"
                          }`}
                          disabled={!isPaused}
                        />
                        <label
                          htmlFor="randomize-tiger"
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
                        className={`w-full ${
                          isPaused
                            ? "bg-gray-700 text-white"
                            : "bg-gray-800 text-gray-500 cursor-not-allowed"
                        } rounded px-2 py-1 text-sm`}
                        disabled={!isPaused}
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
                          settings.players.tiger.settings?.max_time_seconds ||
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
                        className={`w-full ${
                          isPaused
                            ? "bg-gray-700 text-white"
                            : "bg-gray-800 text-gray-500 cursor-not-allowed"
                        } rounded px-2 py-1 text-sm`}
                        disabled={!isPaused}
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
                        className={`w-full ${
                          isPaused
                            ? "bg-gray-700 text-white"
                            : "bg-gray-800 text-gray-500 cursor-not-allowed"
                        } rounded px-2 py-1 text-sm`}
                        disabled={!isPaused}
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
                          settings.players.tiger.settings?.max_rollout_depth ||
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
                        className={`w-full ${
                          isPaused
                            ? "bg-gray-700 text-white"
                            : "bg-gray-800 text-gray-500 cursor-not-allowed"
                        } rounded px-2 py-1 text-sm`}
                        disabled={!isPaused}
                      />
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
                onClick={() => {
                  if (!isPaused) return;
                  setSettings({
                    ...settings,
                    showAdvancedSettings: !settings.showAdvancedSettings,
                  });
                }}
                className={`text-sm ${
                  isPaused
                    ? "text-blue-400 hover:text-blue-300"
                    : "text-gray-500 cursor-not-allowed"
                } flex items-center`}
                disabled={!isPaused}
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

        {/* Time Controls */}
        <div className="space-y-4">
          <h3 className="text-md font-bold text-white border-b border-gray-700 pb-2">
            Time Control
          </h3>

          <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
            {TIME_PRESETS.map((preset) => (
              <button
                key={preset.name}
                className={`p-2 rounded ${
                  settings.selectedPreset.name === preset.name &&
                  !settings.isCustom
                    ? "bg-blue-600 text-white"
                    : isPaused
                    ? "bg-gray-700 text-gray-300 hover:bg-gray-600"
                    : "bg-gray-700 text-gray-500 cursor-not-allowed"
                }`}
                onClick={() => {
                  if (!isPaused) return;
                  setSettings({
                    ...settings,
                    selectedPreset: preset,
                    isCustom: preset.name === "Custom",
                  });
                }}
                disabled={!isPaused}
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
                  className={`w-full ${
                    isPaused
                      ? "bg-gray-700 text-white"
                      : "bg-gray-800 text-gray-500 cursor-not-allowed"
                  } rounded px-3 py-2`}
                  value={settings.customTime.initial / 60}
                  onChange={(e) => {
                    if (!isPaused) return;
                    setSettings({
                      ...settings,
                      customTime: {
                        ...settings.customTime,
                        initial: parseInt(e.target.value) * 60,
                      },
                    });
                  }}
                  disabled={!isPaused}
                />
              </div>

              <div className="flex items-center gap-2">
                <input
                  type="checkbox"
                  id="use-increment"
                  className={`${
                    isPaused ? "bg-gray-700" : "bg-gray-800 cursor-not-allowed"
                  }`}
                  checked={settings.customTime.useIncrement}
                  onChange={(e) => {
                    if (!isPaused) return;
                    setSettings({
                      ...settings,
                      customTime: {
                        ...settings.customTime,
                        useIncrement: e.target.checked,
                      },
                    });
                  }}
                  disabled={!isPaused}
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
                    className={`w-full ${
                      isPaused
                        ? "bg-gray-700 text-white"
                        : "bg-gray-800 text-gray-500 cursor-not-allowed"
                    } rounded px-3 py-2`}
                    value={settings.customTime.increment}
                    onChange={(e) => {
                      if (!isPaused) return;
                      setSettings({
                        ...settings,
                        customTime: {
                          ...settings.customTime,
                          increment: parseInt(e.target.value),
                        },
                      });
                    }}
                    disabled={!isPaused}
                  />
                </div>
              )}
            </div>
          )}
        </div>
      </div>

      {/* Footer with Resume/Apply button */}
      {isPaused && (
        <div className="p-4 border-t border-gray-700">
          <button
            onClick={handleApplyOrResume}
            className={`w-full px-4 py-2 rounded text-sm font-medium transition-colors ${
              hasChanges
                ? "bg-blue-600 hover:bg-blue-700 text-white"
                : "bg-green-600 hover:bg-green-700 text-white"
            }`}
            disabled={gameStatus !== "PLAYING"}
          >
            {hasChanges ? "Apply and Resume" : "Resume"}
          </button>
        </div>
      )}
    </div>
  );
};

export default GameSettingsPanel;
