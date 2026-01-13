import { useState, useCallback } from 'react';

type HistoryState<T> = {
  past: T[];
  present: T;
  future: T[];
};

type SetStateAction<T> = (prevState: T) => T;
type StateUpdater<T> = (action: SetStateAction<T> | T, overwrite?: boolean) => void;
type StateResetter<T> = (newState: T) => void;

const useHistoryState = <T>(initialState: T): [T, StateUpdater<T>, () => void, () => void, StateResetter<T>, boolean, boolean] => {
  const [state, setState] = useState<HistoryState<T>>({
    past: [],
    present: initialState,
    future: [],
  });

  const canUndo = state.past.length > 0;
  const canRedo = state.future.length > 0;

  const updateState: StateUpdater<T> = useCallback((action, overwrite = false) => {
    setState(currentState => {
      const newPresent = typeof action === 'function' ? (action as SetStateAction<T>)(currentState.present) : action;

      if (overwrite) {
          // For actions like dragging, we want to overwrite the present state without creating a new history entry
          return { ...currentState, present: newPresent };
      }
      
      // If the new state is the same as the present, do nothing
      if (JSON.stringify(newPresent) === JSON.stringify(currentState.present)) {
        return currentState;
      }
      
      // Add the new state to history
      const newPast = [...currentState.past, currentState.present];
      return {
        past: newPast,
        present: newPresent,
        future: [], // Clear future when a new action is taken
      };
    });
  }, []);

  const undo = useCallback(() => {
    if (!canUndo) return;
    setState(currentState => {
      const newPresent = currentState.past[currentState.past.length - 1];
      const newPast = currentState.past.slice(0, currentState.past.length - 1);
      // FIX: The original code was spreading currentState.present which was incorrect if `present` is an array.
      const newFuture = [currentState.present, ...currentState.future];
      return {
        past: newPast,
        present: newPresent,
        future: newFuture,
      };
    });
  }, [canUndo]);

  const redo = useCallback(() => {
    if (!canRedo) return;
    setState(currentState => {
      const newPresent = currentState.future[0];
      const newPast = [...currentState.past, currentState.present];
      const newFuture = currentState.future.slice(1);
      return {
        past: newPast,
        present: newPresent,
        future: newFuture,
      };
    });
  }, [canRedo]);

  const resetState: StateResetter<T> = useCallback((newState: T) => {
    setState({
      past: [],
      present: newState,
      future: [],
    });
  }, []);

  return [state.present, updateState, undo, redo, resetState, canUndo, canRedo];
};

export default useHistoryState;