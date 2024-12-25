MDP --> MODEL FREE  (dYNAMIQUE DE L'ENVIRONEMENT EST INCONNU)
    --> MODEL BASED (ENVIRONEMENT BIEN CONNU: La sdynamique de l'env est tres bien connu, (P_transition, Reward, ...))

    --> MODEL BASED:
        - Dynamic programming
        - Policy Iteration

    --> MODEL FREE:
        - Tabular approaches
          - Monte carlos
            - first visit
            - every visite
            - exploring starts
          - Temporal difference 
            - TD(0), TD(lambda)
            - Q-Learning
            - SARSA
            - ESARSA
          (For both we canadd epsilon-greedy)
        - Function approximations
            - Deep Q-Learning

