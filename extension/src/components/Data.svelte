<script lang="ts">
  import type { BiasScore } from "../background";
  import SingleResult from "./SingleResult.svelte";
  import List from "@smui/list";

  export let biasScore: BiasScore = null;
</script>

<List class="demo-list" twoLine nonInteractive>
  {#if biasScore.bias}
    <SingleResult label="Main Score" value={biasScore.bias} />
  {/if}
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
</List>

<style type="text/scss">
</style>
