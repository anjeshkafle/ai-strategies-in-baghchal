// New function to communicate with backend
const fetchBestMove = async (boardState, phase, agent, model = "random") => {
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
  agent,
  model = "random"
) => {
  const isAI = agent?.toLowerCase() !== "human";
  if (!isAI) return null;
  return await fetchBestMove(boardState, phase, agent, model);
};
