// SPDX-License-Identifier: WTFPL
pragma solidity 0.8.38;

/// @title WorstLLM
/// @notice The world's slowest, stupidest, most expensive "L"LM.
/// @dev The engine is hand-written EVM and holds no state. It runs under STATICCALL, so it cannot
/// write anything. This contract has no owner and no upgrade path.
/// @custom:source https://github.com/ReynardoEW/WorstLLM
contract WorstLLM {
    uint256 public constant STATE_SIZE = 16448;

    address public immutable ENGINE;

    /// @notice States that a past generate call produced. Only these can be continued.
    mapping(bytes32 => bool) public committed;
    /// @notice The text of the tx that produced each state.
    mapping(bytes32 => string) public text;

    event Reply(bytes32 indexed parent, bytes32 indexed state, string text);
    event State(bytes32 indexed state, bytes data);

    constructor(address engine) {
        ENGINE = engine;
    }

    /// @notice Generate up to n tokens after the prompt, and stop at the end of the reply.
    /// @param seed Chooses the random draws. The same inputs always give the same text.
    /// @param n Maximum new tokens. Prompt plus new tokens must be at most 64.
    /// @param prompt Token ids, one byte each.
    /// @param prevState The data of a past State event, or empty for a new chat.
    function generate(uint256 seed, uint256 n, bytes calldata prompt, bytes calldata prevState)
        external
        returns (string memory reply, bytes32 state)
    {
        bytes32 parent = _parent(prevState);
        bytes memory data;
        (data, reply) = _run(seed, n, prompt, prevState);
        state = keccak256(data);
        committed[state] = true;
        text[state] = reply;
        emit Reply(parent, state, reply);
        emit State(state, data);
    }

    /// @notice The same run as generate, for free through eth_call. It stores nothing and proves
    /// nothing, so it accepts any state, and generate still rejects the state it returns.
    function preview(uint256 seed, uint256 n, bytes calldata prompt, bytes calldata prevState)
        external
        view
        returns (string memory reply, bytes32 state, bytes memory data)
    {
        (data, reply) = _run(seed, n, prompt, prevState);
        state = keccak256(data);
    }

    function _parent(bytes calldata prevState) private view returns (bytes32 parent) {
        if (prevState.length != 0) {
            parent = keccak256(prevState);
            require(committed[parent], "state not committed");
        }
    }

    function _run(uint256 seed, uint256 n, bytes calldata prompt, bytes calldata prevState)
        private
        view
        returns (bytes memory data, string memory reply)
    {
        require(prevState.length == 0 || prevState.length == STATE_SIZE, "bad state size");
        uint256 flags = 2 | (prevState.length != 0 ? 1 : 0); // bit 1: stop at end of reply, bit 0: has state
        (bool ok, bytes memory out) = ENGINE.staticcall(abi.encodePacked(seed, n, prompt.length, flags, prompt, prevState));
        require(ok, "engine failed");
        uint256 len = out.length - STATE_SIZE;
        reply = new string(len);
        assembly ("memory-safe") {
            mcopy(add(reply, 32), add(add(out, 32), STATE_SIZE), len)
            mstore(out, STATE_SIZE) // the state is the front of the engine's output
        }
        data = out;
    }
}
