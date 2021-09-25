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
    {#if biasScore.predictions?.hatespeech}
      <SingleResult
        label="Communication type"
        value={biasScore.predictions?.hatespeech}
      />
    {/if}
    {#if biasScore.predictions?.hyperpartisan}
      <SingleResult
        label="Political bias"
        value={biasScore.predictions?.hyperpartisan}
      />
    {/if}
    {#if biasScore.predictions?.stance}
      <SingleResult label="Stance" value={biasScore.predictions?.stance} />
    {/if}
  </Card>
</List>

<style type="text/scss">
  :global(.main_card) {
    margin-bottom: 5px;
  }
</style>
