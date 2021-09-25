<script lang="ts">
  import { Item, Text, PrimaryText, SecondaryText } from "@smui/list";
  import Tooltip, { Wrapper } from "@smui/tooltip";

  export let label: string = null;
  export let value: string = null;
</script>

<Item>
  <Text>
    <Wrapper>
      <PrimaryText>
        {label} <i class="fa fa-question" style="font-size:16px;" />
        <Tooltip><slot /></Tooltip>
      </PrimaryText>
    </Wrapper>
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
