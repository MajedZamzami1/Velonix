import React, { useState, useEffect, useRef } from 'react';

function App() {
  const [user, setUser] = useState(null);
  const [question, setQuestion] = useState('');
  const [answer, setAnswer] = useState('');
  const [roles, setRoles] = useState([]);
  const [loading, setLoading] = useState(false);
  const [activeTab, setActiveTab] = useState('chat');
  const [error, setError] = useState(null);
  const [chunks, setChunks] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const chunkListRef = useRef(null);

  useEffect(() => {
    const savedUser = localStorage.getItem('user');
    if (savedUser) {
      setUser(JSON.parse(savedUser));
      fetchRoles();
    }
    // eslint-disable-next-line
  }, []);

  useEffect(() => {
    if (chunkListRef.current) {
      chunkListRef.current.scrollTop = chunkListRef.current.scrollHeight;
    }
  }, [chunks]);

  const handleLogin = async (e) => {
    e.preventDefault();
    const name = e.target.name.value;
    try {
      const response = await fetch('http://localhost:8000/auth', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name })
      });
      const data = await response.json();
      if (response.ok) {
        setUser(data);
        localStorage.setItem('user', JSON.stringify(data));
        fetchRoles();
      } else {
        setError('Authentication failed');
      }
    } catch (error) {
      setError('Login failed');
    }
  };

  const fetchRoles = async () => {
    try {
      const response = await fetch('http://localhost:8000/roles');
      const data = await response.json();
      setRoles(data);
    } catch (error) {
      setError('Failed to fetch roles');
    }
  };

  const handleAskQuestion = async (e) => {
    e.preventDefault();
    if (!question.trim()) return;
    setLoading(true);
    setError(null);
    try {
      const response = await fetch('http://localhost:8000/ask', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          text: question,
          user: { name: user.name }
        })
      });
      const data = await response.json();
      if (response.ok) {
        setAnswer(data.answer);
      } else {
        setError('Failed to get answer');
      }
    } catch (error) {
      setError('Failed to get answer');
    } finally {
      setLoading(false);
    }
  };

  const handleFileUpload = async (e) => {
    e.preventDefault();
    const fileInput = document.getElementById('fileInput');
    const file = fileInput.files[0];
    if (!file) {
      setError('Please select a file');
      return;
    }
    setIsLoading(true);
    setError(null);
    setChunks([]);
    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch('http://localhost:8000/upload-stream', {
        method: 'POST',
        body: formData,
      });
      if (!response.body) {
        setError('No response body');
        setIsLoading(false);
        return;
      }
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';
      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        let lines = buffer.split('\n\n');
        buffer = lines.pop(); // last incomplete event
        for (let line of lines) {
          if (line.startsWith('data: ')) {
            const chunk = JSON.parse(line.replace('data: ', ''));
            setChunks((prev) => [...prev, chunk]);
          }
        }
      }
    } catch (err) {
      setError('Error uploading file: ' + err.message);
    } finally {
      setIsLoading(false);
    }
  };

  const handleRoleUpdate = async (name, finance, hr, it) => {
    setError(null);
    try {
      const response = await fetch('http://localhost:8000/roles', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name, finance, hr, it })
      });
      if (response.ok) {
        fetchRoles();
      } else {
        setError('Failed to update role');
      }
    } catch (error) {
      setError('Failed to update role');
    }
  };

  const handleLogout = () => {
    setUser(null);
    localStorage.removeItem('user');
  };

  if (!user) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-gray-100">
        <div className="bg-white p-8 rounded-lg shadow-md w-full max-w-md">
          <h1 className="text-2xl font-bold mb-6 text-center">Velonix Login</h1>
          {error && (
            <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded mb-4">
              {error}
            </div>
          )}
          <form onSubmit={handleLogin}>
            <input
              type="text"
              name="name"
              placeholder="Enter your name"
              className="w-full p-2 border rounded mb-4"
              required
            />
            <button
              type="submit"
              className="w-full bg-blue-500 text-white p-2 rounded hover:bg-blue-600"
            >
              Login
            </button>
          </form>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-100">
      <nav className="bg-white shadow-md">
        <div className="max-w-7xl mx-auto px-4">
          <div className="flex justify-between h-16">
            <div className="flex items-center">
              <h1 className="text-xl font-bold">Velonix RAG System</h1>
            </div>
            <div className="flex items-center">
              <span className="mr-4">Welcome, {user.name}</span>
              <button
                onClick={handleLogout}
                className="bg-red-500 text-white px-4 py-2 rounded hover:bg-red-600"
              >
                Logout
              </button>
            </div>
          </div>
        </div>
      </nav>
      <div className="max-w-7xl mx-auto px-4 py-8">
        {error && (
          <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded mb-4">
            {error}
          </div>
        )}
        <div className="flex flex-wrap gap-4 mb-6">
          <button
            onClick={() => setActiveTab('chat')}
            className={`px-4 py-2 rounded font-semibold transition ${
              activeTab === 'chat' ? 'bg-blue-500 text-white' : 'bg-gray-200 hover:bg-gray-300'
            }`}
          >
            Chat
          </button>
          <button
            onClick={() => setActiveTab('roles')}
            className={`px-4 py-2 rounded font-semibold transition ${
              activeTab === 'roles' ? 'bg-blue-500 text-white' : 'bg-gray-200 hover:bg-gray-300'
            }`}
          >
            Manage Roles
          </button>
          <button
            onClick={() => setActiveTab('upload')}
            className={`px-4 py-2 rounded font-semibold transition ${
              activeTab === 'upload' ? 'bg-blue-500 text-white' : 'bg-gray-200 hover:bg-gray-300'
            }`}
          >
            Upload Documents
          </button>
        </div>
        {activeTab === 'chat' && (
          <div className="bg-white rounded-lg shadow-md p-6">
            <form onSubmit={handleAskQuestion} className="mb-6">
              <textarea
                value={question}
                onChange={(e) => setQuestion(e.target.value)}
                placeholder="Ask a question..."
                className="w-full p-2 border rounded mb-4 h-32"
                required
              />
              <button
                type="submit"
                disabled={loading}
                className="bg-blue-500 text-white px-6 py-2 rounded hover:bg-blue-600 disabled:bg-gray-400"
              >
                {loading ? 'Getting Answer...' : 'Ask Question'}
              </button>
            </form>
            {answer && (
              <div className="mt-6 p-4 bg-gray-50 rounded">
                <h3 className="font-bold mb-2">Answer:</h3>
                <p>{answer}</p>
              </div>
            )}
          </div>
        )}
        {activeTab === 'roles' && (
          <div className="bg-white rounded-lg shadow-md p-6">
            <h2 className="text-xl font-bold mb-4">Manage Roles</h2>
            <div className="space-y-4">
              {roles.map((role) => (
                <div key={role.name} className="flex items-center space-x-4">
                  <span className="font-medium">{role.name}</span>
                  <div className="flex space-x-2">
                    <label className="flex items-center">
                      <input
                        type="checkbox"
                        checked={role.finance}
                        onChange={(e) => handleRoleUpdate(
                          role.name,
                          e.target.checked,
                          role.hr,
                          role.it
                        )}
                        className="mr-1"
                      />
                      Finance
                    </label>
                    <label className="flex items-center">
                      <input
                        type="checkbox"
                        checked={role.hr}
                        onChange={(e) => handleRoleUpdate(
                          role.name,
                          role.finance,
                          e.target.checked,
                          role.it
                        )}
                        className="mr-1"
                      />
                      HR
                    </label>
                    <label className="flex items-center">
                      <input
                        type="checkbox"
                        checked={role.it}
                        onChange={(e) => handleRoleUpdate(
                          role.name,
                          role.finance,
                          role.hr,
                          e.target.checked
                        )}
                        className="mr-1"
                      />
                      IT
                    </label>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
        {activeTab === 'upload' && (
          <div className="bg-white rounded-lg shadow-md p-6">
            <h2 className="text-xl font-bold mb-4">Upload Documents</h2>
            <form onSubmit={handleFileUpload} className="border-2 border-dashed border-gray-300 rounded-lg p-6 text-center">
              <input
                type="file"
                id="fileInput"
                className="hidden"
                accept=".pdf"
              />
              <label
                htmlFor="fileInput"
                className="cursor-pointer bg-blue-500 text-white px-6 py-2 rounded hover:bg-blue-600"
              >
                Choose File
              </label>
              <p className="mt-2 text-sm text-gray-500">
                Supported format: PDF
              </p>
              <button
                type="submit"
                disabled={isLoading}
                className="mt-4 bg-green-500 text-white px-6 py-2 rounded hover:bg-green-600 disabled:bg-gray-400"
              >
                {isLoading ? 'Processing...' : 'Upload'}
              </button>
            </form>
            {isLoading && (
              <div className="mt-4 text-center">
                <p className="text-gray-600">Processing document...</p>
              </div>
            )}
            {error && (
              <div className="mt-4 bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded">
                {error}
              </div>
            )}
            {chunks.length > 0 && (
              <div className="mt-4">
                <div className="bg-green-100 border border-green-400 text-green-700 px-4 py-3 rounded mb-4">
                  Document processed successfully! Found {chunks.length} chunks.
                </div>
                <h3 className="text-lg font-semibold mb-2">Processed Chunks:</h3>
                <div ref={chunkListRef} className="max-h-96 overflow-y-auto border rounded-lg p-4 bg-gray-50" style={{ minHeight: '200px' }}>
                  {chunks.map((chunk, index) => (
                    <div key={index} className="mb-4 p-3 bg-white rounded-lg shadow flex flex-col gap-2">
                      <div className="flex items-center justify-between mb-1">
                        <span className="font-medium text-gray-700">Chunk {chunk.chunk_number}</span>
                        <div className="flex gap-2">
                          {chunk.tags.finance && (
                            <span className="px-2 py-1 bg-blue-100 text-blue-800 rounded-full text-xs font-semibold">
                              Finance
                            </span>
                          )}
                          {chunk.tags.it && (
                            <span className="px-2 py-1 bg-green-100 text-green-800 rounded-full text-xs font-semibold">
                              IT
                            </span>
                          )}
                          {chunk.tags.hr && (
                            <span className="px-2 py-1 bg-purple-100 text-purple-800 rounded-full text-xs font-semibold">
                              HR
                            </span>
                          )}
                        </div>
                      </div>
                      <p className="text-gray-700 text-sm">{chunk.content}</p>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

export default App;