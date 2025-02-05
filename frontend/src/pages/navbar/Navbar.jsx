import React from "react";
import { Link, useNavigate } from "react-router-dom";

const Navbar = () => {
  const navigate = useNavigate();

  const handleLoginClick = () => {
    navigate("/Login");
  };
  const handleSignupClick = () => {
    navigate("/Signup");
  };
  return (
    <>
      {/* Navbar */}
      <nav className="bg-gray-700 sticky top-0 z-50 p-4 flex flex-wrap items-center justify-between">
        <div className="text-white text-xl md:text-2xl font-bold">
          Smart MED Notes
        </div>
        {/* Navigation Links */}
        <div className="flex flex-wrap space-x-2 md:space-x-4 text-sm md:text-base">
          <Link to="/" className="text-white hover:text-gray-400">
            Home
          </Link>
          <Link to="/Assistant" className="text-white hover:text-gray-400">
            Assistant
          </Link>
          <Link to="/About" className="text-white hover:text-gray-400">
            About Us
          </Link>
          <Link to="/Contact" className="text-white hover:text-gray-400">
            Contact
          </Link>
        </div>
        {/* Buttons */}
        <div className="flex flex-wrap space-x-2 mt-2 md:mt-0">
          <button
            onClick={handleSignupClick}
            className="bg-gray-600 text-white px-3 py-2 text-sm md:text-base rounded hover:bg-gray-400"
          >
            Sign Up
          </button>
          <button
            onClick={handleLoginClick}
            className="bg-gray-600 text-white px-3 py-2 text-sm md:text-base rounded hover:bg-gray-400"
          >
            Log In
          </button>
        </div>
      </nav>
    </>
  );
};

export default Navbar;
