import { useState } from 'react'
import './App.css'

function App() {
  const [goal, setGoal] = useState('')

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    // TODO: Implement goal submission
    console.log('Goal submitted:', goal)
  }

  return (
    <div className="app">
      <header className="app-header">
        <h1>Forge Agent</h1>
        <p>Self-hosted autonomous code agent</p>
      </header>
      <main className="app-main">
        <form onSubmit={handleSubmit} className="goal-form">
          <textarea
            value={goal}
            onChange={(e) => setGoal(e.target.value)}
            placeholder="Describe what you want the agent to do..."
            className="goal-input"
            rows={5}
          />
          <button type="submit" className="submit-button">
            Execute Goal
          </button>
        </form>
      </main>
    </div>
  )
}

export default App

