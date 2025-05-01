<template>
    <form @submit.prevent="onSubmit">
      <!-- Player 1 Inputs -->
      <h2>Player 1</h2>
      <label>Seed:</label>
      <input type="number" v-model.number="player1_seed"><br>
      <label>Rank:</label>
      <input type="number" v-model.number="player1_rank"><br>
      <label>IOC:</label>
      <input type="text" v-model="player1_ioc"><br>
      <label>Age:</label>
      <input type="number" v-model.number="player1_age"><br>
      <label>Height:</label>
      <input type="number" v-model.number="player1_ht"><br>
  
      <!-- Player 2 Inputs -->
      <h2>Player 2</h2>
      <label>Seed:</label>
      <input type="number" v-model.number="player2_seed"><br>
      <label>Rank:</label>
      <input type="number" v-model.number="player2_rank"><br>
      <label>IOC:</label>
      <input type="text" v-model="player2_ioc"><br>
      <label>Age:</label>
      <input type="number" v-model.number="player2_age"><br>
      <label>Height:</label>
      <input type="number" v-model.number="player2_ht"><br>
  
      <!-- General Inputs -->
      <h2>Match Details</h2>
      <label>Surface:</label>
      <input type="text" v-model="surface"><br>
      <label>Tourney Level:</label>
      <input type="text" v-model="tourney_level"><br>
  
      <button type="submit">Predict</button>
    </form>
  </template>
  
  <script>
  export default {
    data() {
      return {
        player1_seed: null,
        player1_rank: null,
        player1_ioc: '',
        player1_age: null,
        player1_ht: null,
  
        player2_seed: null,
        player2_rank: null,
        player2_ioc: '',
        player2_age: null,
        player2_ht: null,
  
        surface: '',
        tourney_level: ''
      };
    },
    methods: {
      onSubmit() {
        const inputData = {
          player1_seed: this.player1_seed,
          player1_rank: this.player1_rank,
          player1_ioc: this.player1_ioc,
          player1_age: this.player1_age,
          player1_ht: this.player1_ht,
  
          player2_seed: this.player2_seed,
          player2_rank: this.player2_rank,
          player2_ioc: this.player2_ioc,
          player2_age: this.player2_age,
          player2_ht: this.player2_ht,
  
          surface: this.surface,
          tourney_level: this.tourney_level
        };
  
        this.$emit('prediction', inputData);
      }
    }
  };
  </script>