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
        label="Language"
        value={biasScore.predictions?.hatespeech.prediction}
      >
        A measure indicating use of offensive and hateful language targeting
        abused communities in the article. Possible values are: not detected,
        offensive or hateful language detected
      </SingleResult>
    {/if}
    {#if !biasScore.predictions?.hyperpartisan?.error}
      <SingleResult
        label="Political bias"
        value={biasScore.predictions?.hyperpartisan?.prediction}
        >A measure indicating political polarization or hyper partisanism of the
        article. It can be either to the left (liberal) or to the right
        (conservative). Possible values are: detected or not detected.
      </SingleResult>
    {/if}
    {#if !biasScore.predictions?.stance?.error}
      <SingleResult
        label="Stance"
        value={biasScore.predictions?.stance?.prediction}
        >A measure assessing relation between article’s title and its content.
        Possible values are: balanced, misleading, complementary and critical.
      </SingleResult>
    {/if}
  </Card>
</List>

<style type="text/scss">
  :global(.main_card) {
    margin-bottom: 5px;
  }
</style>
