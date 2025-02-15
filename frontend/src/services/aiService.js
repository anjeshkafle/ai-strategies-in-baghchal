// New function to communicate with backend
const fetchBestMove = async (
  boardState,
  phase,
  agent,
  model,
  goatsPlaced = 0,
  goatsCaptured = 0
) => {
  try {
    const response = await fetch("http://localhost:8000/get-best-move", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        board: boardState,
        phase,
        agent,
        model,
        goats_placed: goatsPlaced,
        goats_captured: goatsCaptured,
      }),
    });

    if (!response.ok) {
      throw new Error("Network response was not ok");
    }

    return await response.json();
  } catch (error) {
    console.error("Error fetching best move:", error);
    throw error;
  }
};

// Get best move based on the current state and AI model
export const getBestMove = async (
  boardState,
  phase,
  playerConfig,
  currentTurn,
  gameState = {}
) => {
  const isAI = playerConfig?.type?.toLowerCase() === "ai";
  if (!isAI || !playerConfig.model) return null;

  return await fetchBestMove(
    boardState,
    phase,
    currentTurn,
    playerConfig.model,
    gameState.goatsPlaced || 0,
    gameState.goatsCaptured || 0
  );
};
