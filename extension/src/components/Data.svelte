<script lang="ts">
  import type { BiasScore } from "../background";
  import SingleResult from "./SingleResult.svelte";
  import MainBias from "./MainBias.svelte";
  import List from "@smui/list";
  import Card from "@smui/card";

  export let biasScore: BiasScore = null;
</script>

<List class="demo-list" twoLine nonInteractive>
  {#if biasScore.bias}
    <Card padded class="main_card">
      <MainBias bias={biasScore.bias} />
    </Card>
  {/if}
  <Card padded>
    {#if !biasScore.predictions?.hatespeech?.error}
      <SingleResult
        label="Communication type"
        value={biasScore.predictions?.hatespeech.prediction}
      />
    {/if}
    {#if !biasScore.predictions?.hyperpartisan?.error}
      <SingleResult
        label="Political bias"
        value={biasScore.predictions?.hyperpartisan?.prediction}
      />
    {/if}
    {#if !biasScore.predictions?.stance?.error}
      <SingleResult
        label="Stance"
        value={biasScore.predictions?.stance?.prediction}
      />
    {/if}
  </Card>
</List>

<style type="text/scss">
  :global(.main_card) {
    margin-bottom: 5px;
  }
</style>
