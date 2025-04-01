// New function to communicate with backend
const fetchBestMove = async (
  boardState,
  phase,
  agent,
  model,
  gameState = {},
  abortController = null
) => {
  try {
    console.log("Fetching best move:", {
      board: boardState,
      phase,
      agent,
      model,
      goats_placed: gameState.goatsPlaced || 0,
      goats_captured: gameState.goatsCaptured || 0,
      settings: gameState.settings || null,
    });

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
        goats_placed: gameState.goatsPlaced || 0,
        goats_captured: gameState.goatsCaptured || 0,
        settings: gameState.settings || null,
      }),
      signal: abortController?.signal,
    });

    if (!response.ok) {
      throw new Error(`Network response was not ok: ${response.status}`);
    }

    const data = await response.json();
    console.log("Received move from backend:", data);
    return data;
  } catch (error) {
    if (error.name === "AbortError") {
      console.log("Request was aborted");
      return null;
    }
    console.error("Error fetching best move:", error);
    throw error;
  }
};

// Get best move based on the current state and AI model
export const getBestMove = async (
  boardState,
  phase,
  agent,
  model,
  gameState = {},
  abortController = null
) => {
  try {
    return await fetchBestMove(
      boardState,
      phase,
      agent,
      model,
      gameState,
      abortController
    );
  } catch (error) {
    console.error("Error in getBestMove:", error);
    return null;
  }
};
