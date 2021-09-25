<script lang="ts">
  import { Item, Text, PrimaryText, SecondaryText } from "@smui/list";
  import Tooltip, { Wrapper } from "@smui/tooltip";

  export let bias: string = null;
</script>

<Item>
  <Text>
    <Wrapper>
      <PrimaryText>
        Score <i class="fa fa-question" style="font-size:16px;" />
      </PrimaryText>
      <Tooltip>General score calculated based on more in-depth models</Tooltip>
    </Wrapper>
    <SecondaryText>
      <div class="main_result">{bias.toLowerCase()}</div>
    </SecondaryText>
  </Text>
</Item>

<style type="text/scss">
  :global(.main_result) {
    text-transform: capitalize;
  }
</style>
