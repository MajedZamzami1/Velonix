const { useState, useEffect } = React;

function App() {
    const [user, setUser] = useState(null);
    const [question, setQuestion] = useState('');
    const [answer, setAnswer] = useState('');
    const [roles, setRoles] = useState([]);
    const [loading, setLoading] = useState(false);
    const [activeTab, setActiveTab] = useState('chat');
    const [error, setError] = useState(null);

    useEffect(() => {
        // Check if user is already logged in
        const savedUser = localStorage.getItem('user');
        if (savedUser) {
            setUser(JSON.parse(savedUser));
            fetchRoles();
        }
    }, []);

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
            console.error('Login error:', error);
            setError('Login failed');
        }
    };

    const fetchRoles = async () => {
        try {
            const response = await fetch('http://localhost:8000/roles');
            const data = await response.json();
            setRoles(data);
        } catch (error) {
            console.error('Error fetching roles:', error);
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
            console.error('Error asking question:', error);
            setError('Failed to get answer');
        } finally {
            setLoading(false);
        }
    };

    const handleFileUpload = async (e) => {
        const file = e.target.files[0];
        if (!file) return;

        const formData = new FormData();
        formData.append('file', file);

        setLoading(true);
        setError(null);
        try {
            const response = await fetch('http://localhost:8000/upload', {
                method: 'POST',
                body: formData
            });
            const data = await response.json();
            if (response.ok) {
                alert('Document uploaded successfully');
            } else {
                setError('Upload failed');
            }
        } catch (error) {
            console.error('Upload error:', error);
            setError('Upload failed');
        } finally {
            setLoading(false);
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
            console.error('Error updating role:', error);
            setError('Failed to update role');
        }
    };

    const handleLogout = () => {
        setUser(null);
        localStorage.removeItem('user');
    };

    if (!user) {
        return (
            <div className="min-h-screen flex items-center justify-center">
                <div className="bg-white p-8 rounded-lg shadow-md w-96">
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

                <div className="flex space-x-4 mb-6">
                    <button
                        onClick={() => setActiveTab('chat')}
                        className={`px-4 py-2 rounded ${
                            activeTab === 'chat' ? 'bg-blue-500 text-white' : 'bg-gray-200'
                        }`}
                    >
                        Chat
                    </button>
                    <button
                        onClick={() => setActiveTab('roles')}
                        className={`px-4 py-2 rounded ${
                            activeTab === 'roles' ? 'bg-blue-500 text-white' : 'bg-gray-200'
                        }`}
                    >
                        Manage Roles
                    </button>
                    <button
                        onClick={() => setActiveTab('upload')}
                        className={`px-4 py-2 rounded ${
                            activeTab === 'upload' ? 'bg-blue-500 text-white' : 'bg-gray-200'
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
                        <div className="border-2 border-dashed border-gray-300 rounded-lg p-6 text-center">
                            <input
                                type="file"
                                onChange={handleFileUpload}
                                className="hidden"
                                id="file-upload"
                                accept=".pdf,.doc,.docx,.txt"
                            />
                            <label
                                htmlFor="file-upload"
                                className="cursor-pointer bg-blue-500 text-white px-6 py-2 rounded hover:bg-blue-600"
                            >
                                Choose File
                            </label>
                            <p className="mt-2 text-sm text-gray-500">
                                Supported formats: PDF, DOC, DOCX, TXT
                            </p>
                        </div>
                    </div>
                )}
            </div>
        </div>
    );
}

ReactDOM.render(<App />, document.getElementById('root')); 