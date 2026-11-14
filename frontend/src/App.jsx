import React, { useState } from 'react';

export default function App() {
  return (
    <div className="min-h-screen bg-slate-50 text-slate-800 flex items-center justify-center p-6">
      <div className="max-w-lg bg-white rounded-2xl shadow-xl p-8 text-center border border-slate-100">
        <div className="w-16 h-16 bg-blue-600 rounded-2xl mx-auto flex items-center justify-center text-white text-3xl mb-4 shadow-lg">
          ⚕️
        </div>
        <h1 className="text-2xl font-bold text-slate-900 mb-2">HealthRisk AI Platform</h1>
        <p className="text-sm text-slate-500 mb-6">
          The interactive web application is served at root via FastAPI. Please visit <a href="/" className="text-blue-600 font-bold underline">http://localhost:8000/</a> to experience the full React dashboard.
        </p>
      </div>
    </div>
  );
}
