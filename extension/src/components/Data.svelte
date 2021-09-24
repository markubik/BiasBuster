<script lang="ts">
  import type { BiasScore } from "../background";

  export let biasScore: BiasScore;
</script>

{#if biasScore.bias}
  <div>
    <label for="biasScore">Main Score</label>
    <span id="biasScore">{biasScore.bias.toLowerCase()}</span>
  </div>
{/if}

{#if biasScore?.predictions?.hate_speech}
  <div class="single_result">
    <label for="hateSpeech">Hate Speech</label>
    <span id="hateSpeech">
      {biasScore?.predictions?.hate_speech.toLowerCase()}
    </span>
  </div>
{/if}

{#if biasScore?.predictions?.hyperpartizan}
  <div class="single_result">
    <label for="hyperpartizan">Hyperpartizan</label>
    <span id="hyperpartizan">
      {biasScore?.predictions?.hyperpartizan.toLowerCase()}
    </span>
  </div>
{/if}

{#if biasScore?.predictions?.stance}
  <div class="single_result">
    <label for="stance">Stance</label>
    <span id="stance">
      {biasScore?.predictions?.stance.toLowerCase()}
    </span>
  </div>
{/if}

<style type="text/scss">
  .single_result {
    width: 100%;
  }
</style>
