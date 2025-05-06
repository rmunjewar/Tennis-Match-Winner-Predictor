/**
 * Tennis Match Prediction API
 */
const API_BASE_URL = process.env.NODE_ENV === 'development' ? 'http://localhost:5001/api' : '/api';

/**
 * send prediction request to the backend API
 * @param {Object} data - Player and match data for prediction
 * @returns {Promise<Object>} - Prediction result with winner and confidence
 */
export async function getPrediction(data) {
  try {
    const response = await fetch(`${API_BASE_URL}/predict`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(data),
    });

    if (!response.ok) {
      throw new Error(`HTTP error! Status: ${response.status}`);
    }

    const result = await response.json();
    if (result.error) {
      throw new Error(result.error);
    }
    return result;
  } catch (error) {
    console.error("Error fetching prediction:", error);
    if (error instanceof SyntaxError) {
      // If it's a JSON parsing error, try to get the raw text
      try {
        const rawText = await response.text();
        console.error("Raw response text:", rawText);
      } catch (textError) {
        console.error("Error getting raw response text:", textError);
      }
    }
    throw error;
  }
}

/**
 * get information about the model features and importance
 * @returns {Promise<Object>} - Model information
 */
export async function getModelInfo() {
  try {
    const response = await fetch(`${API_BASE_URL}/model-info`);

    if (!response.ok) {
      throw new Error(`HTTP error! Status: ${response.status}`);
    }

    return await response.json();
  } catch (error) {
    console.error("Error fetching model info:", error);
    throw error;
  }
}

/**
 * for development and testing without a backend
 * @param {Object} data - Player and match data
 * @returns {Promise<Object>} - Mock prediction result with simulated analysis
 */
export async function getMockPrediction(data) {
  return new Promise((resolve) => {
    // simulate API delay
    setTimeout(() => {
      // Initialize default values
      const p1Rank = data.player1_rank || 100;
      const p2Rank = data.player2_rank || 100;
      const p1Age = data.player1_age || 25;
      const p2Age = data.player2_age || 25;
      const p1Height = data.player1_ht || 185;
      const p2Height = data.player2_ht || 185;
      const p1Aces = data.player1_ace || 0;
      const p2Aces = data.player2_ace || 0;
      const p1DF = data.player1_df || 0;
      const p2DF = data.player2_df || 0;
      const p1Wins = data.wins_w || 0;
      const p2Wins = data.wins_l || 0;
      const p1Matches = data.matches_played_w || 1;
      const p2Matches = data.matches_played_l || 1;

      // Calculate various factors (all normalized between -1 and 1)
      
      // 1. Ranking Factor (0-0.3 range)
      const rankDiff = Math.abs(p1Rank - p2Rank);
      const rankAdvantage = Math.min(0.3, rankDiff / 200);
      const rankFactor = p1Rank < p2Rank ? rankAdvantage : -rankAdvantage;

      // 2. Age Factor (0-0.15 range)
      // Younger players have advantage, but not too young (experience matters)
      const ageDiff = p1Age - p2Age;
      let ageFactor = 0;
      if (Math.abs(ageDiff) > 2) {
        // Optimal age range is 24-28
        const p1AgeFactor = Math.max(-0.15, Math.min(0.15, (24 - p1Age) / 10));
        const p2AgeFactor = Math.max(-0.15, Math.min(0.15, (24 - p2Age) / 10));
        ageFactor = p1AgeFactor - p2AgeFactor;
      }

      // 3. Surface Factor (0-0.2 range)
      let surfaceFactor = 0;
      if (data.surface) {
        // Surface specialists by country
        const claySpecialists = ["ESP", "ARG", "ITA", "FRA", "BRA"];
        const grassSpecialists = ["GBR", "AUS", "USA", "CAN"];
        const hardSpecialists = ["USA", "RUS", "JPN", "KOR", "CHN"];

        const surfaceWeight = 0.2;
        if (data.surface === "Clay" && claySpecialists.includes(data.player1_ioc)) {
          surfaceFactor = surfaceWeight;
        } else if (data.surface === "Clay" && claySpecialists.includes(data.player2_ioc)) {
          surfaceFactor = -surfaceWeight;
        } else if (data.surface === "Grass" && grassSpecialists.includes(data.player1_ioc)) {
          surfaceFactor = surfaceWeight;
        } else if (data.surface === "Grass" && grassSpecialists.includes(data.player2_ioc)) {
          surfaceFactor = -surfaceWeight;
        } else if (data.surface === "Hard" && hardSpecialists.includes(data.player1_ioc)) {
          surfaceFactor = surfaceWeight * 0.8; // Slightly less impact for hard courts
        } else if (data.surface === "Hard" && hardSpecialists.includes(data.player2_ioc)) {
          surfaceFactor = -surfaceWeight * 0.8;
        }
      }

      // 4. Height Factor (0-0.15 range)
      // Height advantage varies by surface
      const heightDiff = p1Height - p2Height;
      let heightFactor = 0;
      if (Math.abs(heightDiff) > 5) {
        const heightWeight = 0.15;
        if (data.surface === "Grass") {
          // Height matters most on grass
          heightFactor = (heightDiff / 20) * heightWeight;
        } else if (data.surface === "Hard") {
          // Moderate impact on hard courts
          heightFactor = (heightDiff / 25) * heightWeight;
        } else if (data.surface === "Clay") {
          // Least impact on clay
          heightFactor = (heightDiff / 30) * heightWeight;
        }
      }

      // 5. Serve Factor (0-0.15 range)
      // Based on aces and double faults
      const p1ServeQuality = (p1Aces - p1DF * 2) / 10;
      const p2ServeQuality = (p2Aces - p2DF * 2) / 10;
      const serveFactor = Math.max(-0.15, Math.min(0.15, (p1ServeQuality - p2ServeQuality) / 10));

      // 6. Form Factor (0-0.2 range)
      // Based on recent win percentage
      const p1WinRate = p1Wins / p1Matches;
      const p2WinRate = p2Wins / p2Matches;
      const formFactor = Math.max(-0.2, Math.min(0.2, (p1WinRate - p2WinRate) * 2));

      // Calculate total factor (ranges from -1 to 1)
      const totalFactor = (
        rankFactor * 0.3 +      // 30% weight
        ageFactor * 0.15 +      // 15% weight
        surfaceFactor * 0.2 +   // 20% weight
        heightFactor * 0.15 +   // 15% weight
        serveFactor * 0.1 +     // 10% weight
        formFactor * 0.1        // 10% weight
      );

      // Convert to probability (0.5 means even match)
      let probability = 0.5 + (totalFactor / 2);

      // Clamp to valid probability range
      probability = Math.max(0.1, Math.min(0.9, probability));

      // Determine winner and confidence
      const winner = probability > 0.5 ? 1 : 2;
      const confidence = winner === 1 ? probability : 1 - probability;

      // Generate factors for analysis
      const factors = [];

      // Only include significant factors
      if (Math.abs(rankFactor) > 0.05) {
        const betterPlayer = rankFactor > 0 ? 1 : 2;
        factors.push(
          `Player ${betterPlayer}'s higher ranking (${Math.abs(p1Rank - p2Rank)} positions difference)`
        );
      }

      if (Math.abs(ageFactor) > 0.05) {
        const youngerPlayer = p1Age < p2Age ? 1 : 2;
        factors.push(
          `Player ${youngerPlayer}'s age advantage (${Math.abs(p1Age - p2Age).toFixed(1)} years difference)`
        );
      }

      if (Math.abs(surfaceFactor) > 0.05) {
        const betterId = surfaceFactor > 0 ? 1 : 2;
        factors.push(
          `Player ${betterId}'s experience on ${data.surface} courts`
        );
      }

      if (Math.abs(heightFactor) > 0.05) {
        const tallerId = heightFactor > 0 ? 1 : 2;
        factors.push(
          `Player ${tallerId}'s height advantage (${Math.abs(heightDiff)} cm) on ${data.surface}`
        );
      }

      if (Math.abs(serveFactor) > 0.05) {
        const betterServer = serveFactor > 0 ? 1 : 2;
        factors.push(
          `Player ${betterServer}'s superior serving (${Math.abs(p1Aces - p2Aces)} more aces)`
        );
      }

      if (Math.abs(formFactor) > 0.05) {
        const betterForm = formFactor > 0 ? 1 : 2;
        factors.push(
          `Player ${betterForm}'s better recent form (${Math.abs((p1WinRate - p2WinRate) * 100).toFixed(1)}% higher win rate)`
        );
      }

      // Create response object
      const result = {
        winner: winner,
        confidence: parseFloat((confidence * 100).toFixed(1)),
        analysis: {
          summary: `Player ${winner} is ${
            confidence > 0.7
              ? "strongly"
              : confidence > 0.6
              ? "moderately"
              : "slightly"
          } favored to win this match with ${(confidence * 100).toFixed(1)}% confidence.`,
          factors: factors,
          disclaimer: "This is a mock prediction for testing purposes only.",
          detailed_analysis: {
            ranking_impact: parseFloat((rankFactor * 100).toFixed(1)),
            age_impact: parseFloat((ageFactor * 100).toFixed(1)),
            surface_impact: parseFloat((surfaceFactor * 100).toFixed(1)),
            height_impact: parseFloat((heightFactor * 100).toFixed(1)),
            serve_impact: parseFloat((serveFactor * 100).toFixed(1)),
            form_impact: parseFloat((formFactor * 100).toFixed(1))
          }
        }
      };

      resolve(result);
    }, 800); // network delay
  });
}

/**
 * get country name from IOC code
 * @param {string} iocCode - IOC country code (3-letter)
 * @returns {string} - Full country name
 */
export function getCountryFromIOC(iocCode) {
  const countries = {
    USA: "United States",
    GBR: "Great Britain",
    ESP: "Spain",
    SRB: "Serbia",
    RUS: "Russia",
    SUI: "Switzerland",
    GER: "Germany",
    ITA: "Italy",
    FRA: "France",
    AUT: "Austria",
    JPN: "Japan",
    ARG: "Argentina",
    AUS: "Australia",
    CAN: "Canada",
    CRO: "Croatia",
    POL: "Poland",
    GRE: "Greece",
    CZE: "Czech Republic",
    BLR: "Belarus",
    NOR: "Norway",
    RSA: "South Africa",
    BUL: "Bulgaria",
    BEL: "Belgium",
    NED: "Netherlands",
    KAZ: "Kazakhstan",
    SWE: "Sweden",
  };

  return countries[iocCode] || iocCode;
}

/**
 * get surface display name
 * @param {string} surfaceCode - Surface code from API
 * @returns {string} - User-friendly surface name
 */
export function getSurfaceDisplayName(surfaceCode) {
  const surfaces = {
    H: "Hard Court",
    C: "Clay Court",
    G: "Grass Court",
    Hard: "Hard Court",
    Clay: "Clay Court",
    Grass: "Grass Court",
    Carpet: "Carpet Court",
  };

  return surfaces[surfaceCode] || surfaceCode;
}

/**
 *  tournament level display name
 * @param {string} levelCode - Tournament level code
 * @returns {string} - User-friendly tournament level
 */
export function getTournamentLevelName(levelCode) {
  const levels = {
    G: "Grand Slam",
    M: "Masters 1000",
    A: "ATP 500",
    B: "ATP 250",
    F: "Tour Finals",
    D: "Davis Cup",
  };

  return levels[levelCode] || levelCode;
}

/**
 * format player data for display
 * @param {Object} data - raw player data
 * @param {number} playerNum - player number (1 or 2)
 * @returns {Object} - formmatted player data
 */
export function formatPlayerData(data, playerNum) {
  const prefix = `player${playerNum}_`;

  return {
    name: data[`${prefix}name`] || `Player ${playerNum}`,
    rank: data[`${prefix}rank`] || "Unranked",
    seed:
      data[`${prefix}seed`] && data[`${prefix}seed`] !== 999
        ? data[`${prefix}seed`]
        : null,
    country: getCountryFromIOC(data[`${prefix}ioc`]),
    countryCode: data[`${prefix}ioc`],
    age: data[`${prefix}age`]
      ? parseFloat(data[`${prefix}age`]).toFixed(1)
      : null,
    height: data[`${prefix}ht`] ? `${data[`${prefix}ht`]} cm` : null,
  };
}
