import React from 'react';
import { screen, waitFor } from '@testing-library/react';
import fetchMock from 'mocks/fetchMock';
import userEvent from '@testing-library/user-event';
import { render } from '../../../setupTests';
import DJClientContext from '../../providers/djclient';
import NodeListActions from '../NodeListActions';

describe('<NodeListActions />', () => {
  beforeEach(() => {
    fetchMock.resetMocks();
    vi.clearAllMocks();
    window.scrollTo = vi.fn();
  });

  const renderElement = (djClient, nodeType) => {
    return render(
      <DJClientContext.Provider value={djClient}>
        <NodeListActions nodeName="default.hard_hat" nodeType={nodeType} />
      </DJClientContext.Provider>,
    );
  };

  const initializeMockDJClient = () => {
    return {
      DataJunctionAPI: {
        deactivate: vi.fn(),
        downstreams: vi.fn(),
      },
    };
  };

  it('deletes a node when clicked', async () => {
    global.confirm = () => true;
    const mockDjClient = initializeMockDJClient();
    mockDjClient.DataJunctionAPI.deactivate.mockReturnValue({
      status: 204,
      json: { name: 'source.warehouse.schema.some_table' },
    });

    renderElement(mockDjClient);

    await userEvent.click(screen.getByRole('button'));

    await waitFor(() => {
      expect(mockDjClient.DataJunctionAPI.deactivate).toBeCalled();
      expect(mockDjClient.DataJunctionAPI.deactivate).toBeCalledWith(
        'default.hard_hat',
      );
    });
    await waitFor(() => {
      expect(
        screen.getByText('Successfully deleted node default.hard_hat'),
      ).toBeInTheDocument();
    });
  }, 60000);

  it('skips a node deletion during confirm', async () => {
    global.confirm = () => false;
    const mockDjClient = initializeMockDJClient();
    mockDjClient.DataJunctionAPI.deactivate.mockReturnValue({
      status: 204,
      json: { name: 'source.warehouse.schema.some_table' },
    });

    renderElement(mockDjClient);

    await userEvent.click(screen.getByRole('button'));

    await waitFor(() => {
      expect(mockDjClient.DataJunctionAPI.deactivate).not.toBeCalled();
    });
  }, 60000);

  it('fail deleting a node when clicked', async () => {
    global.confirm = () => true;
    const mockDjClient = initializeMockDJClient();
    mockDjClient.DataJunctionAPI.deactivate.mockReturnValue({
      status: 777,
      json: { message: 'source.warehouse.schema.some_table' },
    });

    renderElement(mockDjClient);

    await userEvent.click(screen.getByRole('button'));

    await waitFor(() => {
      expect(mockDjClient.DataJunctionAPI.deactivate).toBeCalled();
      expect(mockDjClient.DataJunctionAPI.deactivate).toBeCalledWith(
        'default.hard_hat',
      );
    });
    expect(
      screen.getByText('source.warehouse.schema.some_table'),
    ).toBeInTheDocument();
  }, 60000);

  it('disables deleting a source node with dependents', async () => {
    const mockDjClient = initializeMockDJClient();
    mockDjClient.DataJunctionAPI.downstreams.mockResolvedValue([
      { name: 'default.repair_orders' },
      { name: 'default.num_repair_orders' },
    ]);

    renderElement(mockDjClient, 'source');

    await waitFor(() => {
      expect(screen.getByRole('button')).toBeDisabled();
    });
    expect(mockDjClient.DataJunctionAPI.downstreams).toBeCalledWith(
      'default.hard_hat',
    );
    expect(screen.getByRole('button').getAttribute('title')).toEqual(
      '2 node(s) depend on this source',
    );

    await userEvent.click(screen.getByRole('button'));
    expect(mockDjClient.DataJunctionAPI.deactivate).not.toBeCalled();
  }, 60000);

  it('allows deleting a source node without dependents', async () => {
    const mockDjClient = initializeMockDJClient();
    mockDjClient.DataJunctionAPI.downstreams.mockResolvedValue([]);

    renderElement(mockDjClient, 'source');

    await waitFor(() => {
      expect(mockDjClient.DataJunctionAPI.downstreams).toBeCalled();
    });
    expect(screen.getByRole('button')).toBeEnabled();
    expect(screen.getByRole('button').getAttribute('title')).toEqual(null);
  }, 60000);

  it('ignores a downstreams lookup failure', async () => {
    const mockDjClient = initializeMockDJClient();
    mockDjClient.DataJunctionAPI.downstreams.mockResolvedValue({
      message: 'Node not found',
    });

    renderElement(mockDjClient, 'source');

    await waitFor(() => {
      expect(mockDjClient.DataJunctionAPI.downstreams).toBeCalled();
    });
    expect(screen.getByRole('button')).toBeEnabled();
  }, 60000);
});
