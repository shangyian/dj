import * as React from 'react';
import { useContext, useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import DJClientContext from '../../providers/djclient';
import NodeStatus from '../NodePage/NodeStatus';
import NodeListActions from '../../components/NodeListActions';
import LoadingIcon from '../../icons/LoadingIcon';
import { formatRelativeTime } from '../../utils/date';

import 'styles/settings.css';

export function MyWorkspacePage() {
  const djClient = useContext(DJClientContext).DataJunctionAPI;

  const [currentUser, setCurrentUser] = useState(null);
  const [userLoading, setUserLoading] = useState(true);

  // Data sections with individual loading states
  const [ownedNodes, setOwnedNodes] = useState([]);
  const [watchedNodes, setWatchedNodes] = useState([]);
  const [recentlyEdited, setRecentlyEdited] = useState([]);
  const [myNodesLoading, setMyNodesLoading] = useState(true);

  const [notifications, setNotifications] = useState([]);
  const [notificationsLoading, setNotificationsLoading] = useState(true);

  const [collections, setCollections] = useState([]);
  const [collectionsLoading, setCollectionsLoading] = useState(true);

  // Actionable items (all contribute to Needs Attention)
  const [nodesMissingDescription, setNodesMissingDescription] = useState([]);
  const [invalidNodes, setInvalidNodes] = useState([]);
  const [staleDrafts, setStaleDrafts] = useState([]);
  const [orphanedDimensions, setOrphanedDimensions] = useState([]);
  const [needsAttentionLoading, setNeedsAttentionLoading] = useState(true);

  const [materializedNodes, setMaterializedNodes] = useState([]);
  const [materializationsLoading, setMaterializationsLoading] = useState(true);

  // Check if user's personal namespace exists
  const [hasPersonalNamespace, setHasPersonalNamespace] = useState(null);
  const [namespaceLoading, setNamespaceLoading] = useState(true);

  // Fetch user first, then kick off all other fetches independently
  useEffect(() => {
    const fetchUser = async () => {
      try {
        const user = await djClient.whoami();
        setCurrentUser(user);
      } catch (error) {
        console.error('Error fetching user:', error);
      }
      setUserLoading(false);
    };
    fetchUser();
  }, [djClient]);

  // Fetch owned nodes + watched nodes + recently edited (combined into "My Nodes")
  useEffect(() => {
    if (!currentUser?.username) return;
    const fetchData = async () => {
      try {
        // Fetch owned nodes (high limit for accurate stats)
        const ownedData = await djClient.getWorkspaceOwnedNodes(currentUser.username, 5000);
        const owned = ownedData?.data?.findNodesPaginated?.edges?.map(e => e.node) || [];
        setOwnedNodes(owned);

        // Fetch watched nodes (notification subscriptions)
        const subscriptions = await djClient.getNotificationPreferences({ entity_type: 'node' });
        const watchedNodeNames = (subscriptions || []).map(s => s.entity_name);
        
        if (watchedNodeNames.length > 0) {
          const watchedData = await djClient.getNodesByNames(watchedNodeNames);
          setWatchedNodes(watchedData || []);
        } else {
          setWatchedNodes([]);
        }

        // Fetch recently edited nodes (high limit for accurate stats)
        const editedData = await djClient.getWorkspaceRecentlyEdited(currentUser.username, 5000);
        const edited = editedData?.data?.findNodesPaginated?.edges?.map(e => e.node) || [];
        setRecentlyEdited(edited);
      } catch (error) {
        console.error('Error fetching my nodes:', error);
      }
      setMyNodesLoading(false);
    };
    fetchData();
  }, [djClient, currentUser?.username]);

  // Fetch notifications (recent activity on subscribed nodes)
  useEffect(() => {
    if (!currentUser?.username) return;
    const fetchData = async () => {
      try {
        const history = await djClient.getSubscribedHistory(50);
        
        // Enrich with node info
        const nodeNames = Array.from(new Set((history || []).map(h => h.entity_name)));
        let nodeInfoMap = {};
        if (nodeNames.length > 0) {
          const nodes = await djClient.getNodesByNames(nodeNames);
          nodeInfoMap = Object.fromEntries((nodes || []).map(n => [n.name, n]));
        }
        
        const enriched = (history || []).map(entry => ({
          ...entry,
          node_type: nodeInfoMap[entry.entity_name]?.type,
          display_name: nodeInfoMap[entry.entity_name]?.current?.displayName,
        }));
        
        setNotifications(enriched);
      } catch (error) {
        console.error('Error fetching notifications:', error);
      }
      setNotificationsLoading(false);
    };
    fetchData();
  }, [djClient, currentUser?.username]);

  // Fetch collections
  useEffect(() => {
    if (!currentUser?.username) return;
    const fetchData = async () => {
      try {
        const data = await djClient.getWorkspaceCollections(currentUser.username);
        setCollections(data?.data?.listCollections || []);
      } catch (error) {
        console.error('Error fetching collections:', error);
      }
      setCollectionsLoading(false);
    };
    fetchData();
  }, [djClient, currentUser?.username]);

  // Check if user's personal namespace exists
  useEffect(() => {
    if (!currentUser?.username) return;
    const checkNamespace = async () => {
      try {
        // Extract username without domain for namespace
        const usernameForNamespace = currentUser.username.split('@')[0];
        const personalNamespace = `users.${usernameForNamespace}`;
        const namespaces = await djClient.namespaces();
        const exists = namespaces.some(ns => ns.namespace === personalNamespace);
        setHasPersonalNamespace(exists);
      } catch (error) {
        console.error('Error checking namespace:', error);
        setHasPersonalNamespace(false);
      }
      setNamespaceLoading(false);
    };
    checkNamespace();
  }, [djClient, currentUser?.username]);

  // Fetch needs attention items (invalid, missing desc, stale drafts, orphaned dimensions)
  useEffect(() => {
    if (!currentUser?.username) return;
    const fetchData = async () => {
      try {
        const [missingDescData, invalidData, draftData, orphanedData] = await Promise.all([
          djClient.getWorkspaceNodesMissingDescription(currentUser.username, 5000),
          djClient.getWorkspaceInvalidNodes(currentUser.username, 5000),
          djClient.getWorkspaceDraftNodes(currentUser.username, 5000),
          djClient.getWorkspaceOrphanedDimensions(currentUser.username, 5000),
        ]);

        setNodesMissingDescription(
          missingDescData?.data?.findNodesPaginated?.edges?.map(e => e.node) || [],
        );
        setInvalidNodes(
          invalidData?.data?.findNodesPaginated?.edges?.map(e => e.node) || [],
        );
        setOrphanedDimensions(
          orphanedData?.data?.findNodesPaginated?.edges?.map(e => e.node) || [],
        );

        // Filter drafts older than 7 days
        const sevenDaysAgo = new Date();
        sevenDaysAgo.setDate(sevenDaysAgo.getDate() - 7);
        const allDrafts =
          draftData?.data?.findNodesPaginated?.edges?.map(e => e.node) || [];
        const stale = allDrafts.filter(node => {
          const updatedAt = new Date(node.current?.updatedAt);
          return updatedAt < sevenDaysAgo;
        });
        setStaleDrafts(stale);
      } catch (error) {
        console.error('Error fetching needs attention items:', error);
      }
      setNeedsAttentionLoading(false);
    };
    fetchData();
  }, [djClient, currentUser?.username]);

  // Fetch materializations
  useEffect(() => {
    if (!currentUser?.username) return;
    const fetchData = async () => {
      try {
        const data = await djClient.getWorkspaceMaterializations(currentUser.username, 5000);
        setMaterializedNodes(
          data?.data?.findNodesPaginated?.edges?.map(e => e.node) || [],
        );
      } catch (error) {
        console.error('Error fetching materializations:', error);
      }
      setMaterializationsLoading(false);
    };
    fetchData();
  }, [djClient, currentUser?.username]);

  // Filter stale materializations (> 72 hours old)
  const staleMaterializations = materializedNodes.filter(node => {
    const validThroughTs = node.current?.availability?.validThroughTs;
    if (!validThroughTs) return false; // Pending ones aren't "stale"
    const hoursSinceUpdate = (Date.now() - validThroughTs) / (1000 * 60 * 60);
    return hoursSinceUpdate > 72;
  });

  const hasActionableItems =
    nodesMissingDescription.length > 0 ||
    invalidNodes.length > 0 ||
    staleDrafts.length > 0 ||
    staleMaterializations.length > 0 ||
    orphanedDimensions.length > 0;

  // Personal namespace for the user
  const usernameForNamespace = currentUser?.username?.split('@')[0] || '';
  const personalNamespace = `users.${usernameForNamespace}`;

  if (userLoading) {
    return (
      <div className="settings-page" style={{ padding: '1.5rem 2rem' }}>
        <h1 className="settings-title">My Workspace</h1>
        <div style={{ textAlign: 'center', padding: '3rem' }}>
          <LoadingIcon />
        </div>
      </div>
    );
  }

  // Calculate stats
  return (
    <div className="settings-page" style={{ padding: '1.5rem 2rem' }}>
      <h1 className="settings-title" style={{ marginBottom: '1rem' }}>
        My Workspace
      </h1>

      {/* Row 1: My Nodes (2/3) + Notifications (1/3) */}
      <div
        style={{
          display: 'flex',
          gap: '1.5rem',
          marginBottom: '1.5rem',
        }}
      >
        <div style={{ flex: 2 }}>
          <MyNodesSection
            ownedNodes={ownedNodes}
            watchedNodes={watchedNodes}
            recentlyEdited={recentlyEdited}
            username={currentUser?.username}
            loading={myNodesLoading}
          />
        </div>
        <div style={{ flex: 1 }}>
          <NotificationsSection
            notifications={notifications}
            username={currentUser?.username}
            loading={notificationsLoading}
          />
        </div>
      </div>

      {/* Row 2: Needs Attention + Materializations + Collections */}
      <div
        style={{
          display: 'flex',
          gap: '1.5rem',
          marginBottom: '1.5rem',
        }}
      >
        <div style={{ flex: 1, minWidth: 0, overflow: 'hidden' }}>
          <NeedsAttentionSection
            nodesMissingDescription={nodesMissingDescription}
            invalidNodes={invalidNodes}
            staleDrafts={staleDrafts}
            staleMaterializations={staleMaterializations}
            orphanedDimensions={orphanedDimensions}
            username={currentUser?.username}
            hasItems={hasActionableItems}
            loading={needsAttentionLoading || materializationsLoading}
            personalNamespace={personalNamespace}
            hasPersonalNamespace={hasPersonalNamespace}
            namespaceLoading={namespaceLoading}
          />
        </div>
        <div style={{ flex: 1 }}>
          <MaterializationsSection nodes={materializedNodes} loading={materializationsLoading} />
        </div>
        <div style={{ flex: 1 }}>
          <CollectionsSection collections={collections} loading={collectionsLoading} />
        </div>
      </div>
    </div>
  );
}

// Notifications Section - matches NotificationBell dropdown styling
function NotificationsSection({ notifications, username, loading }) {
  return (
    <section>
      <div className="section-title-row">
        <h2 className="settings-section-title">🔔 Notifications</h2>
        <Link to="/notifications" style={{ fontSize: '13px' }}>
          View All →
        </Link>
      </div>
      <div className="settings-card" style={{ padding: '0', height: '350px', overflowY: 'auto' }}>
        {loading ? (
          <div style={{ textAlign: 'center', padding: '1rem' }}>
            <LoadingIcon /> 
          </div>
        ) : notifications.length > 0 ? (
          <div className="notifications-list">
            {notifications.slice(0, 15).map((entry, index) => {
              const version = entry.details?.version;
              const href = version
                ? `/nodes/${entry.entity_name}/revisions/${version}`
                : `/nodes/${entry.entity_name}/history`;
              return (
                <a key={entry.id || index} className="notification-item" href={href}>
                  <span className="notification-node">
                    <span className="notification-title">
                      {entry.display_name || entry.entity_name}
                      {version && (
                        <span className="badge version">{version}</span>
                      )}
                    </span>
                    {entry.display_name && (
                      <span className="notification-entity">
                        {entry.entity_name}
                      </span>
                    )}
                  </span>
                  <span className="notification-meta">
                    {entry.node_type && (
                      <span
                        className={`node_type__${entry.node_type.toLowerCase()} badge node_type`}
                      >
                        {entry.node_type.toUpperCase()}
                      </span>
                    )}
                    {entry.activity_type}d by{' '}
                    <span style={{ color: entry.user === username ? 'var(--primary-color, #4a90d9)' : '#333' }}>
                      {entry.user === username ? 'you' : entry.user?.split('@')[0]}
                    </span>{' '}
                    · {formatRelativeTime(entry.created_at)}
                  </span>
                </a>
              );
            })}
          </div>
        ) : (
          <div style={{ padding: '1rem', textAlign: 'center', color: '#666', fontSize: '12px' }}>
            <div style={{ fontSize: '24px', marginBottom: '0.5rem' }}>🔔</div>
            <p>No notifications yet.</p>
            <p style={{ fontSize: '11px', marginTop: '0.5rem' }}>
              Watch nodes to get notified of changes.
            </p>
          </div>
        )}
      </div>
    </section>
  );
}

// Needs Attention Section with single-line categories
function NeedsAttentionSection({
  nodesMissingDescription,
  invalidNodes,
  staleDrafts,
  staleMaterializations,
  orphanedDimensions,
  username,
  hasItems,
  loading,
  personalNamespace,
  hasPersonalNamespace,
  namespaceLoading,
}) {
  const categories = [
    {
      id: 'invalid',
      icon: '❌',
      label: 'Invalid',
      nodes: invalidNodes,
      viewAllLink: `/?ownedBy=${username}&statuses=INVALID`,
    },
    {
      id: 'stale-drafts',
      icon: '⏰',
      label: 'Stale Drafts',
      nodes: staleDrafts,
      viewAllLink: `/?ownedBy=${username}&mode=DRAFT`,
    },
    {
      id: 'stale-materializations',
      icon: '📦',
      label: 'Stale Materializations',
      nodes: staleMaterializations,
      viewAllLink: `/?ownedBy=${username}&hasMaterialization=true`,
    },
    {
      id: 'no-description',
      icon: '📝',
      label: 'No Description',
      nodes: nodesMissingDescription,
      viewAllLink: `/?ownedBy=${username}&missingDescription=true`,
    },
    {
      id: 'orphaned-dimensions',
      icon: '🔗',
      label: 'Orphaned Dimensions',
      nodes: orphanedDimensions,
      viewAllLink: `/?ownedBy=${username}&orphanedDimension=true`,
    },
  ];

  return (
    <section style={{ minWidth: 0, width: '100%' }}>
      <h2 className="settings-section-title">⚠️ Needs Attention</h2>
      <div style={{ height: '350px', overflowY: 'auto' }}>
        {loading ? (
          <div className="settings-card" style={{ textAlign: 'center', padding: '1rem' }}>
            <LoadingIcon />
          </div>
        ) : (
          <div style={{ display: 'flex', flexDirection: 'column', width: '100%', gap: '0.5rem' }}>
            {categories.map((cat) => (
              <div key={cat.id} className="settings-card" style={{ padding: '0.5rem 0.75rem', minWidth: 0 }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.3rem' }}>
                  <span style={{ fontSize: '11px', fontWeight: '600', color: '#555' }}>
                    {cat.icon} {cat.label}
                    <span style={{ color: cat.nodes.length > 0 ? '#dc3545' : '#666', marginLeft: '4px' }}>
                      ({cat.nodes.length})
                    </span>
                  </span>
                  {cat.nodes.length > 0 && (
                    <Link to={cat.viewAllLink} style={{ fontSize: '10px' }}>
                      View all →
                    </Link>
                  )}
                </div>
                {cat.nodes.length > 0 ? (
                  <div style={{ display: 'flex', gap: '0.3rem', overflow: 'hidden' }}>
                    {cat.nodes.slice(0, 10).map(node => (
                      <a
                        key={node.name}
                        href={`/nodes/${node.name}`}
                        style={{
                          display: 'inline-flex',
                          alignItems: 'center',
                          gap: '3px',
                          padding: '2px 6px',
                          fontSize: '10px',
                          border: '1px solid var(--border-color, #ddd)',
                          borderRadius: '3px',
                          textDecoration: 'none',
                          color: 'inherit',
                          backgroundColor: 'var(--card-bg, #f8f9fa)',
                          whiteSpace: 'nowrap',
                          flexShrink: 0,
                        }}
                      >
                        <span
                          className={`node_type__${node.type.toLowerCase()} badge node_type`}
                          style={{ fontSize: '7px', padding: '1px 2px' }}
                        >
                          {node.type.charAt(0)}
                        </span>
                        {node.current?.displayName || node.name.split('.').pop()}
                      </a>
                    ))}
                  </div>
                ) : (
                  <div style={{ fontSize: '10px', color: '#28a745' }}>
                    ✓ All good!
                  </div>
                )}
              </div>
            ))}
            {/* Personal namespace prompt if missing */}
            {!namespaceLoading && !hasPersonalNamespace && (
              <div
                style={{
                  padding: '0.75rem',
                  backgroundColor: 'var(--card-bg, #f8f9fa)',
                  border: '1px dashed var(--border-color, #dee2e6)',
                  borderRadius: '6px',
                  textAlign: 'center',
                }}
              >
                <div style={{ fontSize: '16px', marginBottom: '0.25rem' }}>📁</div>
                <div style={{ fontSize: '11px', fontWeight: '500', marginBottom: '0.25rem' }}>
                  Set up your namespace
                </div>
                <p style={{ fontSize: '10px', color: '#666', marginBottom: '0.5rem' }}>
                  Create <code style={{ backgroundColor: '#e9ecef', padding: '1px 4px', borderRadius: '3px', fontSize: '9px' }}>{personalNamespace}</code>
                </p>
                <Link
                  to={`/namespaces/${personalNamespace}`}
                  style={{
                    display: 'inline-block',
                    padding: '3px 8px',
                    fontSize: '10px',
                    backgroundColor: '#28a745',
                    color: '#fff',
                    borderRadius: '4px',
                    textDecoration: 'none',
                  }}
                >
                  Create →
                </Link>
              </div>
            )}
          </div>
        )}
      </div>
    </section>
  );
}

// My Nodes Section (owned + watched, with tabs)
function MyNodesSection({ ownedNodes, watchedNodes, recentlyEdited, username, loading }) {
  const [activeTab, setActiveTab] = React.useState('owned');
  
  const ownedNames = new Set(ownedNodes.map(n => n.name));
  const watchedOnly = watchedNodes.filter(n => !ownedNames.has(n.name));
  
  const allMyNodeNames = new Set([...ownedNames, ...watchedNodes.map(n => n.name)]);
  const editedOnly = recentlyEdited.filter(n => !allMyNodeNames.has(n.name));
  
  const getDisplayNodes = () => {
    switch (activeTab) {
      case 'owned': return ownedNodes;
      case 'watched': return watchedOnly;
      case 'edited': return recentlyEdited;
      default: return ownedNodes;
    }
  };
  const displayNodes = getDisplayNodes();
  
  const hasAnyContent = ownedNodes.length > 0 || watchedOnly.length > 0 || recentlyEdited.length > 0;

  const maxDisplay = 8;

  return (
    <section>
      <div className="section-title-row">
        <h2 className="settings-section-title">👤 My Nodes</h2>
        <Link to={`/?ownedBy=${username}`} style={{ fontSize: '13px' }}>
          View All →
        </Link>
      </div>
      <div className="settings-card" style={{ padding: '0.25rem 0.75rem', height: '350px', overflowY: 'auto' }}>
        {loading ? (
          <div style={{ textAlign: 'center', padding: '1rem' }}>
            <LoadingIcon />
          </div>
        ) : hasAnyContent ? (
          <>
            {/* Tabs */}
            <div style={{ display: 'flex', gap: '0.5rem', marginBottom: '0.5rem', paddingTop: '0.5rem' }}>
              <button
                onClick={() => setActiveTab('owned')}
                style={{
                  padding: '4px 10px',
                  fontSize: '11px',
                  border: 'none',
                  borderRadius: '4px',
                  cursor: 'pointer',
                  backgroundColor: activeTab === 'owned' ? 'var(--primary-color, #4a90d9)' : '#e9ecef',
                  color: activeTab === 'owned' ? '#fff' : '#495057',
                }}
              >
                Owned ({ownedNodes.length})
              </button>
              <button
                onClick={() => setActiveTab('watched')}
                style={{
                  padding: '4px 10px',
                  fontSize: '11px',
                  border: 'none',
                  borderRadius: '4px',
                  cursor: 'pointer',
                  backgroundColor: activeTab === 'watched' ? 'var(--primary-color, #4a90d9)' : '#e9ecef',
                  color: activeTab === 'watched' ? '#fff' : '#495057',
                }}
              >
                Watched ({watchedOnly.length})
              </button>
              <button
                onClick={() => setActiveTab('edited')}
                style={{
                  padding: '4px 10px',
                  fontSize: '11px',
                  border: 'none',
                  borderRadius: '4px',
                  cursor: 'pointer',
                  backgroundColor: activeTab === 'edited' ? 'var(--primary-color, #4a90d9)' : '#e9ecef',
                  color: activeTab === 'edited' ? '#fff' : '#495057',
                }}
              >
                Recent Edits ({recentlyEdited.length})
              </button>
            </div>
            {displayNodes.length > 0 ? (
              <>
                <NodeList nodes={displayNodes.slice(0, maxDisplay)} showUpdatedAt={true} />
                {displayNodes.length > maxDisplay && (
                  <div style={{ textAlign: 'center', padding: '0.5rem', fontSize: '12px', color: '#666' }}>
                    +{displayNodes.length - maxDisplay} more
                  </div>
                )}
              </>
            ) : (
              <div style={{ padding: '1rem', textAlign: 'center', color: '#666', fontSize: '12px' }}>
                {activeTab === 'owned' && 'No owned nodes'}
                {activeTab === 'watched' && 'No watched nodes'}
                {activeTab === 'edited' && 'No recent edits'}
              </div>
            )}
          </>
        ) : (
          <div style={{ padding: '0.75rem 0' }}>
            <p style={{ fontSize: '12px', color: '#666', marginBottom: '0.75rem' }}>
              No nodes yet.
            </p>
            <div style={{ display: 'flex', gap: '0.75rem' }}>
              <div
                style={{
                  flex: 1,
                  padding: '0.75rem',
                  backgroundColor: 'var(--card-bg, #f8f9fa)',
                  border: '1px dashed var(--border-color, #dee2e6)',
                  borderRadius: '6px',
                  textAlign: 'center',
                }}
              >
                <div style={{ fontSize: '16px', marginBottom: '0.25rem' }}>➕</div>
                <div style={{ fontSize: '11px', fontWeight: '500', marginBottom: '0.25rem' }}>
                  Create a node
                </div>
                <p style={{ fontSize: '10px', color: '#666', marginBottom: '0.5rem' }}>
                  Build your data model
                </p>
                <Link
                  to="/create/source"
                  style={{
                    display: 'inline-block',
                    padding: '3px 8px',
                    fontSize: '10px',
                    backgroundColor: 'var(--primary-color, #4a90d9)',
                    color: '#fff',
                    borderRadius: '4px',
                    textDecoration: 'none',
                  }}
                >
                  Create →
                </Link>
              </div>
              <div
                style={{
                  flex: 1,
                  padding: '0.75rem',
                  backgroundColor: 'var(--card-bg, #f8f9fa)',
                  border: '1px dashed var(--border-color, #dee2e6)',
                  borderRadius: '6px',
                  textAlign: 'center',
                }}
              >
                <div style={{ fontSize: '16px', marginBottom: '0.25rem' }}>👤</div>
                <div style={{ fontSize: '11px', fontWeight: '500', marginBottom: '0.25rem' }}>
                  Claim ownership
                </div>
                <p style={{ fontSize: '10px', color: '#666', marginBottom: '0.5rem' }}>
                  Add yourself as owner
                </p>
                <Link
                  to="/"
                  style={{
                    display: 'inline-block',
                    padding: '3px 8px',
                    fontSize: '10px',
                    backgroundColor: '#6c757d',
                    color: '#fff',
                    borderRadius: '4px',
                    textDecoration: 'none',
                  }}
                >
                  Browse →
                </Link>
              </div>
            </div>
          </div>
        )}
      </div>
    </section>
  );
}

// Collections Section
function CollectionsSection({ collections, loading }) {
  return (
    <section>
      <div className="section-title-row">
        <h2 className="settings-section-title">📁 My Collections</h2>
        <Link to="/collections" style={{ fontSize: '13px' }}>
          View All →
        </Link>
      </div>
      <div
        className="settings-card"
        style={{ padding: '0.75rem 1rem', height: '350px', overflowY: 'auto' }}
      >
        {loading ? (
          <div style={{ textAlign: 'center', padding: '1rem' }}>
            <LoadingIcon />
          </div>
        ) : collections.length > 0 ? (
          <div
            style={{
              display: 'flex',
              flexWrap: 'wrap',
              gap: '0.5rem',
            }}
          >
            {collections.slice(0, 6).map(collection => (
              <a
                key={collection.name}
                href={`/collections/${collection.name}`}
                style={{
                  display: 'block',
                  padding: '0.75rem 1rem',
                  border: '1px solid var(--border-color, #e0e0e0)',
                  borderRadius: '6px',
                  textDecoration: 'none',
                  color: 'inherit',
                  transition: 'all 0.15s ease',
                  backgroundColor: 'var(--card-bg, #fff)',
                  maxWidth: '20rem',
                  width: '100%',
                }}
                onMouseEnter={e => {
                  e.currentTarget.style.borderColor = 'var(--primary-color, #007bff)';
                  e.currentTarget.style.boxShadow = '0 2px 8px rgba(0,0,0,0.08)';
                }}
                onMouseLeave={e => {
                  e.currentTarget.style.borderColor = 'var(--border-color, #e0e0e0)';
                  e.currentTarget.style.boxShadow = 'none';
                }}
              >
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <span style={{ fontWeight: '500' }}>{collection.name}</span>
                  <span className="badge bg-secondary-soft rounded-pill" style={{ fontSize: '11px' }}>
                    {collection.nodeCount} nodes
                  </span>
                </div>
                {collection.description && (
                  <div style={{ fontSize: '12px', color: '#888', marginTop: '4px' }}>
                    {collection.description}
                  </div>
                )}
              </a>
            ))}
            {collections.length > 6 && (
              <div style={{ alignSelf: 'center', fontSize: '12px', color: '#666', padding: '0.5rem' }}>
                +{collections.length - 6} more
              </div>
            )}
          </div>
        ) : (
          <div style={{ padding: '0' }}>
            <p style={{ fontSize: '12px', color: '#666', marginBottom: '0.75rem' }}>
              No collections yet.
            </p>
            <div
              style={{
                padding: '0.75rem',
                backgroundColor: 'var(--card-bg, #f8f9fa)',
                border: '1px dashed var(--border-color, #dee2e6)',
                borderRadius: '6px',
                textAlign: 'center',
              }}
            >
              <div style={{ fontSize: '16px', marginBottom: '0.25rem' }}>📁</div>
              <div style={{ fontSize: '11px', fontWeight: '500', marginBottom: '0.25rem' }}>
                Create a collection
              </div>
              <p style={{ fontSize: '10px', color: '#666', marginBottom: '0.5rem' }}>
                Group related nodes together
              </p>
              <Link
                to="/collections"
                style={{
                  display: 'inline-block',
                  padding: '3px 8px',
                  fontSize: '10px',
                  backgroundColor: 'var(--primary-color, #4a90d9)',
                  color: '#fff',
                  borderRadius: '4px',
                  textDecoration: 'none',
                }}
              >
                Create →
              </Link>
            </div>
          </div>
        )}
      </div>
    </section>
  );
}

// Materializations Section
function MaterializationsSection({ nodes, loading }) {
  const sortedNodes = [...nodes].sort((a, b) => {
    const aTs = a.current?.availability?.validThroughTs;
    const bTs = b.current?.availability?.validThroughTs;
    if (!aTs && !bTs) return 0;
    if (!aTs) return 1;
    if (!bTs) return -1;
    return bTs - aTs;
  });

  const getAvailabilityStatus = availability => {
    if (!availability) {
      return { icon: '⏳', text: 'Pending', color: '#6c757d' };
    }
    const validThrough = availability.validThroughTs ? new Date(availability.validThroughTs) : null;
    const now = new Date();
    const hoursSinceUpdate = validThrough ? (now - validThrough) / (1000 * 60 * 60) : null;
    if (!validThrough) {
      return { icon: '⏳', text: 'Pending', color: '#6c757d' };
    } else if (hoursSinceUpdate <= 24) {
      return { icon: '🟢', text: formatTimeAgo(validThrough), color: '#28a745' };
    } else if (hoursSinceUpdate <= 72) {
      return { icon: '🟡', text: formatTimeAgo(validThrough), color: '#ffc107' };
    } else {
      return { icon: '🔴', text: formatTimeAgo(validThrough), color: '#dc3545' };
    }
  };

  const formatTimeAgo = date => {
    const now = new Date();
    const diffMs = now - date;
    const diffHours = Math.floor(diffMs / (1000 * 60 * 60));
    const diffDays = Math.floor(diffHours / 24);
    if (diffHours < 1) return 'just now';
    if (diffHours < 24) return `${diffHours}h ago`;
    if (diffDays === 1) return 'yesterday';
    return `${diffDays}d ago`;
  };

  const maxDisplay = 5;

  return (
    <section>
      <div className="section-title-row">
        <h2 className="settings-section-title">📦 Materializations</h2>
        <Link to="/?hasMaterialization=true" style={{ fontSize: '13px' }}>
          View All →
        </Link>
      </div>
      <div className="settings-card" style={{ padding: '0.75rem 1rem', height: '350px', overflowY: 'auto' }}>
        {loading ? (
          <div style={{ textAlign: 'center', padding: '1rem' }}>
            <LoadingIcon />
          </div>
        ) : sortedNodes.length > 0 ? (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
            {sortedNodes.slice(0, maxDisplay).map(node => {
              const status = getAvailabilityStatus(node.current?.availability);
              return (
                <div
                  key={node.name}
                  style={{
                    padding: '0.5rem',
                    border: '1px solid var(--border-color, #e0e0e0)',
                    borderRadius: '4px',
                    backgroundColor: 'var(--card-bg, #fff)',
                  }}
                >
                  <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '4px' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                      <a href={`/nodes/${node.name}`} style={{ fontSize: '12px', fontWeight: '500' }}>
                        {node.current?.displayName || node.name.split('.').pop()}
                      </a>
                      <span
                        className={`node_type__${node.type.toLowerCase()} badge node_type`}
                        style={{ fontSize: '8px', padding: '1px 3px' }}
                      >
                        {node.type.charAt(0)}
                      </span>
                    </div>
                    <span style={{ fontSize: '10px', color: status.color }}>
                      {status.icon} {status.text}
                    </span>
                  </div>
                  <div style={{ fontSize: '10px', color: '#666' }}>
                    {node.current?.materializations?.map(mat => (
                      <span key={mat.name} style={{ marginRight: '8px' }}>
                        🕐 {mat.schedule || 'No schedule'}
                      </span>
                    ))}
                    {node.current?.availability?.table && (
                      <span style={{ color: '#888' }}>→ {node.current.availability.table}</span>
                    )}
                  </div>
                </div>
              );
            })}
            {sortedNodes.length > maxDisplay && (
              <div style={{ textAlign: 'center', padding: '0.5rem', fontSize: '12px', color: '#666' }}>
                +{sortedNodes.length - maxDisplay} more
              </div>
            )}
          </div>
        ) : (
          <div style={{ padding: '0' }}>
            <p style={{ fontSize: '12px', color: '#666', marginBottom: '0.75rem' }}>
              No materializations configured.
            </p>
            <div
              style={{
                padding: '0.75rem',
                backgroundColor: 'var(--card-bg, #f8f9fa)',
                border: '1px dashed var(--border-color, #dee2e6)',
                borderRadius: '6px',
                textAlign: 'center',
              }}
            >
              <div style={{ fontSize: '16px', marginBottom: '0.25rem' }}>📦</div>
              <div style={{ fontSize: '11px', fontWeight: '500', marginBottom: '0.25rem' }}>
                Materialize a node
              </div>
              <p style={{ fontSize: '10px', color: '#666', marginBottom: '0.5rem' }}>
                Speed up queries with cached data
              </p>
              <Link
                to="/"
                style={{
                  display: 'inline-block',
                  padding: '3px 8px',
                  fontSize: '10px',
                  backgroundColor: 'var(--primary-color, #4a90d9)',
                  color: '#fff',
                  borderRadius: '4px',
                  textDecoration: 'none',
                }}
              >
                Browse nodes →
              </Link>
            </div>
          </div>
        )}
      </div>
    </section>
  );
}

// Node List Component
function NodeList({ nodes, showUpdatedAt }) {
  const formatDateTime = dateStr => {
    const date = new Date(dateStr);
    return date.toLocaleDateString('en-US', { 
      month: 'short', 
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    });
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
      {nodes.map(node => (
        <div
          key={node.name}
          className="subscription-item"
          style={{
            display: 'flex',
            alignItems: 'flex-start',
            justifyContent: 'space-between',
            padding: '0.5rem 0',
            borderBottom: '1px solid var(--border-color, #eee)',
          }}
        >
          <div style={{ flex: 1, minWidth: 0 }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.25rem' }}>
              <a
                href={`/nodes/${node.name}`}
                style={{
                  fontSize: '14px',
                  fontWeight: '500',
                  textDecoration: 'none',
                }}
              >
                {node.current?.displayName || node.name.split('.').pop()}
              </a>
              <span
                className={`node_type__${node.type?.toLowerCase()} badge node_type`}
                style={{ fontSize: '8px', height: '22px' }}
              >
                {node.type}
              </span>
              <span style={{ transform: 'scale(0.8)', transformOrigin: 'left center' }}>
                <NodeStatus node={node} revalidate={false} />
              </span>
            </div>
            <div style={{ fontSize: '12px', color: '#888' }}>
              {node.name}
            </div>
          </div>
          <div style={{ display: 'flex', alignItems: 'flex-start', gap: '0.5rem', flexShrink: 0 }}>
            {showUpdatedAt && node.current?.updatedAt && (
              <span style={{ fontSize: '12px', color: '#888' }}>
                {formatDateTime(node.current.updatedAt)}
              </span>
            )}
            <NodeListActions nodeName={node.name} />
          </div>
        </div>
      ))}
    </div>
  );
}
