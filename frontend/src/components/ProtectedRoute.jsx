import { Navigate } from "react-router-dom";
import { useGameStore } from "../stores/gameStore";

const ProtectedRoute = ({ children }) => {
  const isInitialized = useGameStore((state) => state.isInitialized);

  if (!isInitialized) {
    return <Navigate to="/" replace />;
  }

  return children;
};

export default ProtectedRoute;
