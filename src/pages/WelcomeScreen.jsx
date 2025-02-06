import React from "react";
import { useNavigate } from "react-router-dom";

const WelcomeScreen = () => {
  const navigate = useNavigate();

  return (
    <div className="fixed inset-0 w-full h-full flex flex-col items-center justify-center bg-gradient-to-b from-blue-50 to-gray-100">
      <div className="text-center space-y-8">
        <h1 className="text-5xl font-bold text-gray-800 mb-2">
          Welcome to Baghchal AI
        </h1>
        <p className="text-gray-600 text-lg mb-8">
          Experience the traditional Nepali board game powered by AI
        </p>
        <button
          className="px-8 py-4 bg-blue-600 text-white text-lg font-semibold rounded-lg 
                     shadow-lg hover:bg-blue-700 transform transition-all duration-200 
                     hover:scale-105 active:scale-95"
          onClick={() => navigate("/game")}
        >
          Start Playing
        </button>
      </div>
    </div>
  );
};

export default WelcomeScreen;
