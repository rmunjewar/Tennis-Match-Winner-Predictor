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
      // more sophisticated logic that takes into account different factors
      const p1Rank = data.player1_rank || 100;
      const p2Rank = data.player2_rank || 100;

      // calc rank advantage (0-0.3 range)
      const rankDiff = Math.abs(p1Rank - p2Rank);
      const rankAdvantage = Math.min(0.3, rankDiff / 200);
      const rankFactor = p1Rank < p2Rank ? rankAdvantage : -rankAdvantage;

      // age factor (younger players have slight advantage, 0-0.1 range)
      const p1Age = data.player1_age || 25;
      const p2Age = data.player2_age || 25;
      const ageDiff = Math.abs(p1Age - p2Age);
      const ageFactor = ageDiff > 5 ? (p1Age < p2Age ? 0.1 : -0.1) : 0;

      // surface factor based on surface preference (simplified)
      let surfaceFactor = 0;
      if (data.surface === "Clay") {
        // clay specialists (like Spanish and South American players)
        if (data.player1_ioc === "ESP" || data.player1_ioc === "ARG") {
          surfaceFactor = 0.15;
        } else if (data.player2_ioc === "ESP" || data.player2_ioc === "ARG") {
          surfaceFactor = -0.15;
        }
      } else if (data.surface === "Grass") {
        // grass specialists (like British and Australian players)
        if (data.player1_ioc === "GBR" || data.player1_ioc === "AUS") {
          surfaceFactor = 0.15;
        } else if (data.player2_ioc === "GBR" || data.player2_ioc === "AUS") {
          surfaceFactor = -0.15;
        }
      }

      // height factor (taller players have advantage on faster surfaces)
      const p1Height = data.player1_ht || 185;
      const p2Height = data.player2_ht || 185;
      const heightDiff = p1Height - p2Height;
      let heightFactor = 0;

      if (Math.abs(heightDiff) > 5) {
        if (data.surface === "Grass" || data.surface === "Hard") {
          heightFactor = heightDiff > 0 ? 0.1 : -0.1;
        }
      }

      // calc total factor (ranges from -0.65 to 0.65)
      const totalFactor = rankFactor + ageFactor + surfaceFactor + heightFactor;

      // convert to probability (0.5 means even match, above 0.5 means player 1 has advantage)
      let probability = 0.5 + totalFactor;

      // clamp to valid probability range
      probability = Math.max(0.1, Math.min(0.9, probability));

      // determine winner and confidence
      const winner = probability > 0.5 ? 1 : 2;
      const confidence = winner === 1 ? probability : 1 - probability;

      // generate factors for analysis
      const factors = [];

      // onyl include significant factors
      if (Math.abs(rankFactor) > 0.05) {
        const betterPlayer = rankFactor > 0 ? 1 : 2;
        factors.push(
          `Player ${betterPlayer}'s higher ranking (${Math.abs(
            p1Rank - p2Rank
          )} positions difference)`
        );
      }

      if (Math.abs(ageFactor) > 0.05) {
        const youngerPlayer = p1Age < p2Age ? 1 : 2;
        factors.push(
          `Player ${youngerPlayer} is younger (${Math.abs(
            p1Age - p2Age
          ).toFixed(1)} years difference)`
        );
      }

      if (Math.abs(surfaceFactor) > 0.05) {
        const betterId = surfaceFactor > 0 ? 1 : 2;
        factors.push(
          `Player ${betterId}'s preference for ${data.surface} courts`
        );
      }

      if (Math.abs(heightFactor) > 0.05) {
        const tallerId = heightFactor > 0 ? 1 : 2;
        factors.push(
          `Player ${tallerId}'s height advantage (${Math.abs(heightDiff)} cm)`
        );
      }

      // create response object
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
          } favored to win this match with ${(confidence * 100).toFixed(
            1
          )}% confidence.`,
          factors: factors,
          disclaimer: "this a mock prediction for testing purposes only.",
        },
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
