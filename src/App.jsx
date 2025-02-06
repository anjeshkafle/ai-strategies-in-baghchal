import "./App.css";
import React from "react";
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import WelcomeScreen from "./pages/WelcomeScreen";
import GameScreen from "./pages/GameScreen";

const App = () => {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<WelcomeScreen />} />
        <Route path="/game" element={<GameScreen />} />
      </Routes>
    </Router>
  );
};

export default App;
