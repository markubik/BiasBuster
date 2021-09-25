<script lang="ts">
  import type { Bias } from "../background";
  import type { Message } from "../content_script";

  import Data from "./Data.svelte";
  import Header from "./Header.svelte";
  import Error from "./Error.svelte";

  let biasScore: Bias = null;
  let isOk: boolean = true;

  chrome.tabs.query(
    { active: true, lastFocusedWindow: true },
    (tabs: { url?: string }[]) => {
      var url = tabs[0].url;
      chrome.runtime.sendMessage({ type: "POPUP_INITIALIZED", url });
    }
  );

  chrome.runtime.onMessage.addListener((message: Message) => {
    if (message.type === "SEND_DATA") {
      biasScore = message.data;
    }
    if (message.type === "ERROR") {
      console.log("error");
      console.log(message.data);
      isOk = false;
    }
  });
</script>

<div class="container">
  <Header />
  {#if !isOk}
    <Error />
  {/if}
  {#if biasScore && isOk}
    <Data {biasScore} />
  {/if}
</div>

<style>
  :global(body) {
    background-color: #5cdb95;
  }
  .container {
    border-radius: 10px;
    min-width: 250px;
  }
</style>
