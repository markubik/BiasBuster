<script lang="ts">
  import { Item, Text, PrimaryText, SecondaryText } from "@smui/list";
  import Tooltip, { Wrapper } from "@smui/tooltip";

  export let label: string = null;
  export let value: string = null;
</script>

<Item>
  <Text>
    <PrimaryText>
      <Wrapper>
        <span>{label}</span>
        <Tooltip>Fajny tooltip</Tooltip>
      </Wrapper>
    </PrimaryText>
    <SecondaryText>
      <div class="single_result">{value.toLowerCase()}</div>
    </SecondaryText>
  </Text>
</Item>

<style type="text/scss">
  :global(.mdc-deprecated-list--two-line .mdc-deprecated-list-item) {
    height: 72px;
  }
  :global(.single_result) {
    text-transform: capitalize;
  }
</style>
