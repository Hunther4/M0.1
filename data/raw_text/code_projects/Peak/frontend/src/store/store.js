import { create } from "zustand"
import { api } from "../api/client"

// Slice imports
import { createSkillsSlice } from "./slices/skills"
import { createProfileSlice } from "./slices/profile"
import { createBooksSlice } from "./slices/books"
import { createMentalSlice } from "./slices/mental"
import { createAISlice } from "./slices/ai"
import { createMemoryGameSlice } from "./slices/memoryGame"
import { createMathGameSlice } from "./slices/mathGame"
import { createIQGameSlice } from "./slices/iqGame"

export const useStore = create((set, get) => ({
  // Compose all slices — each returns flat state keys + actions
  ...createSkillsSlice(set, get),
  ...createProfileSlice(set, get),
  ...createBooksSlice(set, get),
  ...createMentalSlice(set, get),
  ...createAISlice(set, get),
  ...createMemoryGameSlice(set, get),
  ...createMathGameSlice(set, get),
  ...createIQGameSlice(set, get),

  // Dual N-Back consolidation helper (delegates to cognitive API)
  consolidateDualNBack: async (cognitiveSessionId) => {
    return await api.cognitive.consolidate(cognitiveSessionId)
  },

  // Reset store to initial state — used by tests
  __unsafe_reset__: () => {
    set({
      // Skills + Sessions
      skills: [],
      sessions: [],
      summary: null,
      groupedSummary: [],
      timeline: [],
      loading: false,
      error: null,
      // Profile
      profile: null,
      profileLoading: true,
      // Books
      books: [],
      booksLoading: false,
      isIndexing: false,
      indexingProgress: 0,
      // Mental
      mentalReps: [],
      challenges: [],
      pendingChallenges: 0,
      generatingRep: false,
      generatingChallenge: false,
      // AI
      ai_mode: "local",
      available_models: [],
      best_model: null,
      selectedModel: { auto: true },
      // Memory Game
      gameSession: null,
      currentRound: null,
      lastAttempt: null,
      gamePhase: "idle",
      gameHistory: [],
      gameError: null,
      // Math Game
      mathSession: null,
      mathCurrentRound: null,
      mathLastAttempt: null,
      mathPhase: "idle",
      mathHistory: [],
      mathError: null,
      mathStats: null,
      // IQ Game
      iqSession: null,
      iqCurrentRound: null,
      iqLastAttempt: null,
      iqPhase: "idle",
      iqHistory: [],
      iqError: null,
    })
  },
}))