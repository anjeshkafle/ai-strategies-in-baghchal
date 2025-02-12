import React, { useState, useRef, useEffect } from "react";

const CustomSelect = ({ value, onChange, options }) => {
  const [isOpen, setIsOpen] = useState(false);
  const dropdownRef = useRef(null);

  useEffect(() => {
    const handleClickOutside = (event) => {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target)) {
        setIsOpen(false);
      }
    };

    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  const handleSelect = (optionValue) => {
    onChange({ target: { value: optionValue } });
    setIsOpen(false);
  };

  const selectedOption = options.find((opt) => opt.value === value);
  // Filter out the currently selected option
  const availableOptions = options.filter((opt) => opt.value !== value);

  return (
    <div className="relative w-36" ref={dropdownRef}>
      <button
        type="button"
        className="w-full bg-gray-700 text-white rounded-lg px-3 py-2 text-left flex items-center justify-between hover:bg-gray-600 focus:outline-none focus:ring-2 focus:ring-gray-500"
        onClick={() => setIsOpen(!isOpen)}
      >
        <span>{selectedOption?.label}</span>
        <span className="text-gray-300">▼</span>
      </button>

      {isOpen && (
        <div className="absolute z-50 w-full mt-1 bg-gray-700 border border-gray-600 rounded-lg shadow-lg">
          {availableOptions.map((option) => (
            <button
              key={option.value}
              className={`w-full px-3 py-2 text-left hover:bg-gray-600 ${
                option.value === value
                  ? "bg-gray-600 text-white"
                  : "text-gray-300"
              } ${option.value === options[0].value ? "rounded-t-lg" : ""} 
              ${
                option.value === options[options.length - 1].value
                  ? "rounded-b-lg"
                  : ""
              }`}
              onClick={() => handleSelect(option.value)}
            >
              {option.label}
            </button>
          ))}
        </div>
      )}
    </div>
  );
};

export default CustomSelect;
